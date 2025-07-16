import logging, threading
from logging.handlers import RotatingFileHandler
import cv2
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os, glob, time, re, sys
import csv
import pyvips
from openslide import OpenSlide
from tiatoolbox import logger
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.utils.misc import imread
from tiatoolbox.utils.visualization import overlay_prediction_contours
from natsort import natsorted
from collections import Counter
import traceback

log_directory = "./uploads/logs"
os.makedirs(log_directory, exist_ok=True)
log_file_path = os.path.join(log_directory, "prediction.log")

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        RotatingFileHandler(log_file_path, maxBytes=10485760, backupCount=5)
                    ])

def predict(file_id_name):
    """Run the prediction for a given file_id and normalization method."""
    full_id = file_id_name
    logging.info(f"Processing file id: {full_id}")
    
    tile_dir = f"./uploads/{full_id}/cell/"
    save_dir_base = f"./uploads/{full_id}/result/"

    tile_paths = glob.glob(os.path.join(tile_dir, "*.png"))
    tile_paths = natsorted(tile_paths)
    total_tiles = len(tile_paths)

    if not tile_paths:
        logging.warning(f"No tiles found for file id: {full_id}")
        return

    # Record start time
    start_time = time.time()
    print(f"progress: 0.0%", flush=True)

    #os.makedirs(save_dir_base, exist_ok=True)

    #progress monitoring
    progress_stop_event = threading.Event()
    progress_thread = None

    def monitor_progress():
        """Monitor output dir for new .dat files"""
        last_processed = 0

        while not progress_stop_event.is_set():
            try:
                processed_files = glob.glob(os.path.join(save_dir_base, "*.dat"))
                processed_count = len(processed_files)

                if processed_count > last_processed:
                    last_processed = processed_count

                    progress = min(99.0, (processed_count / total_tiles) * 100)

                    #update progress every 1%
                    progress_interval = max(1, total_tiles // 100)

                    if processed_count % progress_interval == 0 or processed_count == total_tiles:
                        print(f"progress: {progress:.1f}%", flush=True)
                        logging.info(f"Processed {processed_count}/{total_tiles} tiles ({progress:.1f}%)")

                if processed_count >= total_tiles:
                    break

            except Exception as e:
                logging.error(f"Error in progress monitoring: {e}")

            time.sleep(0.5) #every 0.5 seconds
    

    try:
        progress_thread = threading.Thread(target=monitor_progress, daemon=True)
        progress_thread.start()

        # Initialize the segmentor
        inst_segmentor = NucleusInstanceSegmentor(
            pretrained_model="hovernet_fast-monusac",
            num_loader_workers=2,
            num_postproc_workers=2,
            batch_size=4, #reduced to 1 for sequential processing
        )

        # Perform segmentation on the tile
        inst_segmentor.predict(
            tile_paths,
            save_dir=save_dir_base,
            mode="tile",
            device='cuda',
            crash_on_exception=True)
        
        progress_stop_event.set()

        if progress_thread and progress_thread.is_alive():
            progress_thread.join(timeout=2)

        final_processed = len(glob.glob(os.path.join(save_dir_base, "*.dat")))
        if final_processed >= total_tiles:
            print(f"progress: 100.0%", flush=True)
        
        # Log checkpoint timer
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        logging.info(f"Prediction time: {minutes} minutes {seconds} seconds")
    
    except Exception as e:
        if progress_thread:
            progress_stop_event.set()
        logging.error(f"Error processing file id {full_id}: {e}")
        return None
    

    logging.info(f"Finished processing file id: {full_id}")
    return True

def cellsCount(file_id_name):
    try:
        print("Counting cells started...")
        ## Overlay image making
        # Load each .dat file collect class count and overlay the image 
        dat_dir = f"./uploads/{file_id_name}/result/"  # Directory containing the .dat files
        overlaid_dir = f"./uploads/{file_id_name}/overlay/" 

        os.makedirs(overlaid_dir, exist_ok=True)
        dat_paths = natsorted(glob.glob(os.path.join(dat_dir, "*.dat")))

        tile_dir = f"./uploads/{file_id_name}/cell/"
        csv_file_path = f"./uploads/{file_id_name}/nucleus_info_{file_id_name}.csv"  # Path for the CSV file

        tile_paths = glob.glob(os.path.join(tile_dir, "*.png"))
        sorted_tile_paths = natsorted(tile_paths)
        tile_paths = sorted_tile_paths

        start_time = time.time()

        total_counts = Counter()

        ## Open CSV file to record nucleus counts
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Tile", "Dat File", "Background", "Epithelial", "Lymphocyte", "Macrophage", "Neutrophil"])
            
            # Loop through all .dat files in the directory
            for i in range(len(tile_paths)):
                # Load the predictions
                tile_preds = joblib.load(dat_paths[i])
                tile_name = os.path.splitext(os.path.basename(tile_paths[i]))[0]
                dat_name = os.path.splitext(os.path.basename(dat_paths[i]))[0]
                #logging.info(f"Tile with dat code: ", tile_name, "@", dat_name)
                logging.info(f"Tile with dat code: {tile_name} @ {dat_name}")

                # Count occurrences of each cell type
                class_counts = Counter()
                for nucleus in tile_preds.values():
                    class_id = nucleus["type"]
                    class_counts[class_id] += 1

                # Update the total counts with the current tile's counts            
                total_counts.update(class_counts)         
                    
                # Record counts in CSV
                writer.writerow([
                    tile_name,
                    dat_name,
                    class_counts.get(0, 0),  #background
                    class_counts.get(1, 0),  # Epithelial
                    class_counts.get(2, 0),  # Lymphocyte
                    class_counts.get(3, 0),  # Macrophage
                    class_counts.get(4, 0),  # Neutrophil
                
                ])
                    
                # Read the corresponding tile image for visualization
                tile_img = imread(tile_paths[i])

                # Create the overlay image
                overlaid_predictions = overlay_prediction_contours(
                    canvas=tile_img,
                    inst_dict=tile_preds,
                    draw_dot=False,
                    type_colours={
                        0: ("Background", (255, 255, 255, 0)), #transparent
                        1: ("Epithelial", (255, 0, 0)),
                        2: ("Lymphocyte", (255, 255, 0)),
                        3: ("Macrophage", (0, 255, 0)),
                        4: ("Neutrophil", (0, 0, 255)),
                    },
                    line_thickness=4,    
                )
                        
                # Save the overlaid image
                overlay_path = f"./uploads/{file_id_name}/overlay/overlay_{tile_name}.png"
                plt.imsave(overlay_path, overlaid_predictions)
            
                # After all tiles are processed, write the total counts to the CSV
            writer.writerow([
                "END",
                "Total",
                total_counts.get(0, 0), #total background
                total_counts.get(1, 0),  # Total Epithelial
                total_counts.get(2, 0),  # Total Lymphocyte
                total_counts.get(3, 0),  # Total Macrophage
                total_counts.get(4, 0),  # Total Neutrophil
            ])   

        print(f"progress: 100.0%", flush=True)

        # Calculate the elapsed time
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time //60)
        seconds = int(elapsed_time % 60)
        print(f"Elapsed time for counting cells: {minutes} minutes {seconds} seconds")

    except Exception as e:
        print(f"Error in cellsCount: {str(e)}")
        import traceback
        traceback.print_exc()
        raise #to be caught by the main

    return file_id_name

def main(file_path):
    norm_method = "Vaha"
    filename = os.path.basename(file_path)
    file_id = os.path.splitext(filename)[0]
    file_id_name = f"{file_id}_{norm_method}"

    #create log file
    time_log_path = f"./uploads/{file_id_name}/predict_log.txt"
    os.makedirs(os.path.dirname(time_log_path), exist_ok=True)

    
    with open(time_log_path, 'a') as time_log_file:
        start_time = time.time()

        #unique run identifier
        run_id = f"{file_id_name}_int{(start_time)}"

        logging.info(f"Startig to process run {run_id} for file: {file_id_name}")

        try:
            result = predict(file_id_name)
            if not result:
                logging.warning(f"Prediction failed for file: {file_id_name}")
                time_log_file.write(f"File ID: {file_id_name}, Start time: {time.ctime(start_time)}")
                return
            
            logging.info(f"Prediction completed successfully for file: {file_id_name}") #if success
                #elapsed_time = time.time() - start_time
                #minutes = int(elapsed_time // 60)
                #seconds = int(elapsed_time % 60)
                
                #logging.info(f"Completed processing for file id: {file_id_name}")

            try:
                cellsCount(file_id_name)

                elapsed_time = time.time() - start_time
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                success_msg = f"File ID: {file_id_name}, Start Time: {time.ctime(start_time)}, " \
                    f"Elapsed Time to predict and cells count: {minutes} minutes {seconds} seconds\n"
                
                logging.info(f"Completed full processing for file: {file_id_name} in {minutes}m {seconds}s")
                time_log_file.write(success_msg)

            except Exception as e:
                elapsed_time = time.time() - start_time
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)

                error_msg = f"File ID: {file_id_name}, Start Time: {time.ctime(start_time)}, " \
                    f"Partial failed process (only cellsCount), Elapsed time: {minutes}m {seconds}s, Error: {str(e)}\n"
                
                logging.error(f"Cell count failed for file {file_id_name}: {str(e)}")
                logging.error(f"Trace: {traceback.format_exc()}")
                time_log_file.write(error_msg)

                '''
                import traceback
                error_details = traceback.format_exc()
                logging.error(f"Error in cellsCount for {file_id_name}: {str(e)}\n{error_details}")
                time_log_file.write(f"File ID: {file_id_name}, cellsCount Exception: {str(e)}\n")'''
            #else:
                #time_log_file.write(f"File ID: {file_id_name}, Processing Failed\n")
        except Exception as exc:
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)

            error_msg = f"File ID: {file_id_name}, Start Time: {time.ctime(start_time)}, " \
                f"Failed overall process, Elapsed time: {minutes}m {seconds}s, Error: {str(e)}\n"
                
            logging.error(f"Processing failed for file {file_id_name}: {str(e)}")
            logging.error(f"Trace: {traceback.format_exc()}")
            time_log_file.write(error_msg)
            

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <file_path> <file_name>")
    else:
        file_path = sys.argv[1]
        main(file_path)