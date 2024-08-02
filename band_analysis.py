import numpy as np
import os
import matplotlib.pyplot as plt 
import click

from utils.util import pad_crop, read_envi_file, find_arrays_with_object, set_seed

@click.command()
@click.option("-D", "--data-dir", type=str, default='data\\Train\\ENVI', help="Path for Data Directory")
@click.option("-T", "--viz-type", type=str, default='cropped', help="Visualize Type")


def main(
    data_dir: str,
    viz_type: str,
) -> None:
    """
    Visualize Script for 3 channel Satellite Image.
    """
    click.secho(message="🚀 Visualizing...", fg="green", nl=True)

    data_dir = data_dir
    viz_type = viz_type
    image_dir = os.path.join(data_dir, "Image")
    mask_dir = os.path.join(data_dir, "Mask")
    band_labels = ['NDWI', 'FAI', 'NDCI']

    set_seed(80)

    if viz_type == 'cropped':
        # Read Image, Mask
        linear_norm_list = pad_crop(read_envi_file(image_dir, True, 'linear_norm'), 224)
        dynamic_norm_list = pad_crop(read_envi_file(image_dir, True, 'dynamic_world_norm'), 224)
        image_list = pad_crop(read_envi_file(image_dir, False, None), 224)
        mask_list = pad_crop(read_envi_file(mask_dir, False, None), 224)
        
        # Random Sampling
        indices = find_arrays_with_object(mask_list)
        print(len(indices))

        np.random.shuffle(indices)
        sample = indices[4]

        # Visualization
        plt.figure(figsize=(50,50))
        cols, rows = 3, 4
        img_np, mask_np = image_list[sample], mask_list[sample]
        linear_norm_np, dynamic_world_norm_np = linear_norm_list[sample], dynamic_norm_list[sample]

        for i in range(cols):
            org_band = img_np[i,:,:]
            linear_norm = linear_norm_np[i,:,:]
            dynamic_world_norm = dynamic_world_norm_np[i,:,:]
            true_mask = mask_np[0,:,:]

            results = [org_band, linear_norm, dynamic_world_norm, true_mask]
            labels = [band_labels[i], 'Linear Normalization', 'Dynamic Normalization', 
                    'True Mask']
            
            for j in range(rows):
                plt.subplot(cols, rows, rows*i+j+1)
                plt.imshow(results[j], cmap='gray')
                plt.axis('off')
                plt.title(labels[j])

    elif viz_type == 'band_histogram':
        # Read Image, Mask
        linear_norm_np = read_envi_file(image_dir, True, 'linear_norm')
        dynamic_world_norm_np = read_envi_file(image_dir, True, 'dynamic_world_norm')
        original_image_np = read_envi_file(image_dir, False, None)
        viz_range = [[-200,-0.01], [-0.3,-0.01], [0.01,1.0]]

        # Calculate Band Histogram and Visualization
        plt.figure(figsize=(90,60))
        cols, rows = 3, 2

        for i in range(cols):
            original_image = original_image_np[i,:,:]

            # Land/Sea Masking with mean value of water
            #band_mean = np.mean(original_image[original_image != -np.min(original_image)])
            #original_image[original_image == 0] = band_mean

            linear_norm = linear_norm_np[i,:,:]
            dynamic_world_norm = dynamic_world_norm_np[i,:,:]
            
            original_image_1d = np.array(original_image).ravel()
            linear_norm_1d = np.array(linear_norm).ravel()
            dynamic_world_norm_1d = np.array(dynamic_world_norm).ravel()

            for j in range(rows):
                plt.subplot(cols, rows, rows*i+j+1)
                if j == 0:
                    plt.hist(original_image_1d, 
                            bins = 500,
                            range = viz_range[i],
                            color = 'red',
                            label = 'original_image',
                            histtype = 'step')             
                    plt.title(band_labels[i])     
                    plt.xlabel('Pixel Values(Reflectance)')
                    plt.ylabel('Frequency')  
                else:
                    plt.hist([linear_norm_1d, dynamic_world_norm_1d], 
                            bins = 1000,
                            range = [0.2,0.8],
                            color = ['green', 'blue'],
                            label = [ 'linear_norm', 'dynamic_world_norm'],
                            histtype = 'step',
                            stacked=True)
                    plt.title('After Band Normalization')
                    plt.xlabel('Normalized Pixel Values')
                    plt.ylabel('Frequency')  
                plt.xlim()
                plt.ylim()
                plt.legend(loc='upper left')
    
    plt.subplots_adjust(left=0.125, bottom=0.1,  right=0.9, top=0.9, wspace=0.2, hspace=0.417)         
    plt.show()
    click.secho(message="🚀 End Visualizing...", fg="red", nl=True)
    plt.close()


if __name__ == "__main__":
    main()