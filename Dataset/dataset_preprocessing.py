from PIL import Image
import os
import numpy as np

def create_dirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_images(image_folder,output_folder,files_extension):
    files_name = os.listdir(image_folder)
    for file in files_name:
        if file.endswith(files_extension):
            im = np.fromfile(image_folder+file, dtype=">i2").reshape((2048, 2048)).astype('float64')
            im -= np.amin(im)
            im /= np.amax(im)
            im = (1-im) * 255
            im = Image.fromarray(im).resize((1024,1024), Image.ANTIALIAS)
            im = im.convert('RGB')
            im.save(output_folder+file[:-4]+'.png')
            print (file+' written') 

def create_mask(mask_folder,output_folder,files_extension):
    seg_mask = ["left lung", 'right lung', 'left clavicle', 'right clavicle', 'heart']
    directory=os.listdir(mask_folder)
    for dirs in directory:
        path = mask_folder+dirs+"/masks/"
        files_name = os.listdir(path+'./'+seg_mask[0])
        for file in files_name:
            if file.endswith(files_extension):
                mask_path = path+seg_mask[0]+'/'+file
                mask = np.array(Image.open(mask_path))
                mask[mask > 0] = 1
                for i in range(len(seg_mask[1:])) :
                    temp = np.array(Image.open(path+seg_mask[i+1]+'/'+file)) 
                    mask[temp > 0] = (i+2)

                mask = Image.fromarray(mask)
                mask.save(output_folder+file[:-4]+'.png')
                print (file+' mask written') 

def print_current(text):
    print ('###################')
    print (text)
    print ('###################')

if __name__ == '__main__':
    
    output_folder='./images/'
    images_folder='./All247images/'
    files_extension='.IMG'
    print_current('Generating images')
    create_dirs(output_folder)
    create_images(images_folder,output_folder,files_extension)

    print_current('Generating masks')
    output_folder='./masks/'
    mask_folder='./scratch/'
    files_extension='.gif'
    create_dirs(output_folder)
    create_mask (mask_folder,output_folder,files_extension)


    print_current('All done !')
