import tifffile
from PIL import Image
import numpy as np
import os, time, csv, logging, sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
from time import sleep
from PIL import Image
#import matplotlib as mpl
#import matplotlib.pyplot as plt
from openslide import OpenSlide
from tiatoolbox import data
from tiatoolbox.tools import stainnorm
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.wsicore import wsireader

log_directory = "/app/uploads/logs"
os.makedirs(log_directory, exist_ok=True)
log_file_path = os.path.join(log_directory, "patching.log")

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        RotatingFileHandler(log_file_path, maxBytes=10485760, backupCount=5)
                    ])


def main(file_path):
    try:
        #file_path = os.path.join(folder_path, f"{file_id}.bif")

        #print(f"Processing file: {file_path}")
        norm_method = "Vaha"
        filename = os.path.basename(file_path)
        file_id = os.path.splitext(filename)[0]
        file_id_name = f"{file_id}_{norm_method}"
        output_dir_blank = os.path.join(f"/app/uploads/{file_id_name}/blank/") #need to change this
        output_dir_cell = os.path.join(f"/app/uploads/{file_id_name}/cell/")
        csv_file_path = os.path.join(f"/app/uploads/{file_id_name}/patches_info_{file_id_name}.csv")
        patch_size = 1024
        threshold_std = 5

        #create directories if not exist
        os.makedirs(output_dir_blank, exist_ok=True)
        os.makedirs(output_dir_cell, exist_ok=True)

        #initialize counters
        level=0
        total_patches = 0
        patches_with_cells = 0
        patches_without_cells = 0

        #print initial progress
        print(f"progress: 0.0%")
        sys.stdout.flush()

        #function to read slides and convert into numpy array
        def read_slide(image_data, x, y, width, height, if_openslide = True):
            if if_openslide:
                region = image_data.read_region((x, y), level, (width, height))
                region = region.convert('RGB')
                region = np.asarray(region)
                return region
            else:
                img_height, img_width, _ = image_data.shape
                # Ensure the requested patch stays within bounds
                width = min(width, img_width - x)  # Adjust width if it exceeds the image width
                height = min(height, img_height - y)  # Adjust height if it exceeds the image height
                    
                region = image_data[y:y + height, x:x + width]
                return np.asarray(region)
        
        def generate_patches(image_data, if_openslide, output_dir_blank, output_dir_cell, patch_size, threshold_std, csv_file_path, total_patches, patches_with_cells, patches_without_cells):
            try:    
                if if_openslide:
                    img_width, img_height = image_data.level_dimensions[0]
                else:
                    img_height, img_width, _ = image_data.shape
                print(f"Image dimensions: {img_width}x{img_height}")
                sys.stdout.flush()

                total_patches_expected = ((img_height // patch_size) + 1) * ((img_width // patch_size) + 1)

                progress_interval = max(1, int(total_patches_expected * 0.01))  # 1% intervals

                #initialize stain normalization
                target_image = data.stain_norm_target()
                stain_normalizer = stainnorm.get_normalizer("Vahadane")
                stain_normalizer.fit(target_image)

                with open(csv_file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Lvl0:", img_width, img_height])
                    writer.writerow(["No.", "X", "Y", "Type"])

                    for y in range(0, img_height, patch_size):
                            for x in range(0, img_width, patch_size):
                                try:
                                    # Adjust the patch size near the edges
                                    slide_patch = read_slide(image_data, x, y, patch_size, patch_size, if_openslide)
                                    total_patches += 1

                                    if total_patches % progress_interval == 0:
                                        #calculate progress
                                        progress = min(99.0, (total_patches / total_patches_expected) * 100)
                                        print(f"progress: {progress:.0f}%", flush=True)
                                        #if total_patches % 10 == 0: #update progress every 50 patches
                                         #   print(f"progress: {progress:.0f}%") 
                                          #  sys.stdout.flush()

                                    patch_std = np.mean(np.std(slide_patch, axis=-1))
                                    patch_filename = f"{file_id_name}_{x}_{y}.png"

                                    #selecting patches
                                    if patch_std > threshold_std:
                                        # Apply normalization on cell image
                                        #target_image = data.stain_norm_target()
                                        #stain_normalizer = stainnorm.get_normalizer("Vahadane")
                                        #stain_normalizer.fit(target_image)

                                        slide_patch = stain_normalizer.transform(slide_patch.copy())
                                        patch_full_path = os.path.join(output_dir_cell, patch_filename)
                                        patches_with_cells += 1
                                        patch_type = "cell"
                                    else:
                                        patch_full_path = os.path.join(output_dir_blank, patch_filename)
                                        patches_without_cells += 1
                                        patch_type = "blank"

                                    Image.fromarray(slide_patch).save(patch_full_path)
                                    #plt.imsave(patch_full_path, slide_patch)
                                    #print(f"Saved patch {total_patches}: {patch_full_path}")

                                    #coordinate records
                                    writer.writerow([total_patches, x, y, patch_type])
                                except Exception as e:
                                    logging.error(f"Error processing patch at {x},{y}: {e}")
                                    continue

                return total_patches, patches_with_cells, patches_without_cells
            except Exception as e:
                logging.error(f"Error in generate_patches: {e}")
                raise
        
        start_time = time.time()

        #if openslide cannot open use tifffile
        # Load the image using tifffile

        if file_path.endswith('.svs') or file_path.endswith('.tif'):
            slide = OpenSlide(file_path)
            if_openslide = True

        elif file_path.endswith('.bif'):
            with tifffile.TiffFile(file_path) as tif:
                slide = tif.pages[2].asarray()
                if_openslide = False
        else:
            raise ValueError("Unsupported file type")

        #with tifffile.TiffFile(file_path) as tif:
            #slide = tif.pages[2].asarray()

        #generate patches and save the patches
        total_patches, patches_with_cells, patches_without_cells = generate_patches(
            slide, if_openslide, output_dir_blank, output_dir_cell, patch_size, threshold_std, csv_file_path,
            total_patches, patches_with_cells, patches_without_cells)
        
        #final progress update
        print(f"progress: 100%")
        sys.stdout.flush()

        

        end_time = time.time()

        elapsed_time = end_time - start_time
        minutes = int(elapsed_time //60)
        seconds = int(elapsed_time % 60)
        #print(f"Elapsed time: {minutes} minutes {seconds} seconds")

        # File Processed: {file_path}
        #File Name: {file_id_name}
        #       Elapsed time: {minutes} minutes {seconds} seconds

        #summary = f"""
        #Total patches generated: {total_patches}
        #Patches with cells: {patches_with_cells}
        #Patches without cells: {patches_without_cells}
        #"""
        #print(summary)

        print(f"Total patches generated: {total_patches}")
        print(f"Patches with cells: {patches_with_cells}")
        print(f"Patches without cells: {patches_without_cells}")
        sys.stdout.flush()

        logging.info(f"Total patches: {total_patches}, with cells: {patches_with_cells}, without cells: {patches_without_cells}")
        logging.info(f"Process completed in {time.time() - start_time:.2f} seconds")

        return True

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

        return False 

            
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python patch.py <filepath>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    main(file_path)