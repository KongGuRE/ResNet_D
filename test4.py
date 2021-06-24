import torchvision
import torchvision.transforms as transforms
import torch
import os
import cv2
import numpy as np

if __name__ == "__main__":
    NG_Crack_data_names = []
    OK_data_names = []
    root_dir = r".\Train"
    for (root, dirs, files) in os.walk(root_dir):
        print(root)
        if root == r".\Train\NG_Crack":
            if len(files) > 0:
                for file_name in files:
                    NG_Crack_data_names.append(r'.\Train\NG_Crack\\' + file_name)
                    print(r'.\Train\NG_Crack\\' + file_name)

        if root == r".\Train\OK":
            if len(files) > 0:
                for file_name in files:
                    OK_data_names.append(r'.\Train\OK\\' + file_name)
                    print(r'.\Train\OK\\' + file_name)

    OK_data_names = OK_data_names
    NG_Crack_data_names = NG_Crack_data_names
    print((len(OK_data_names) + len(NG_Crack_data_names)))
    RGBimg = np.zeros(((len(OK_data_names) + len(NG_Crack_data_names)), 3, 32, 32))  # 624, 3, 512, 512
    labelimgOK = np.zeros((len(OK_data_names)))
    labelimgNG = np.ones((len(NG_Crack_data_names)))
    labelimg = np.zeros((len(OK_data_names) + len(NG_Crack_data_names)))

    labelimg[0:(len(OK_data_names))] = labelimgOK
    labelimg[(len(labelimgOK)):] = labelimgNG



    for File_idx, imgFile in enumerate(OK_data_names):
        # print("File_idx: ", File_idx)
        coloredImg = cv2.imread(imgFile)
        #IMG Downsampling
        coloredImg = cv2.pyrDown(coloredImg)
        for a in range(3):
            coloredImg = cv2.pyrDown(coloredImg)

        # cv2.imshow('gray_image', coloredImg)
        # cv2.waitKey(0)
        # print(coloredImg.shape)
        # exit()

        # IMG split
        b, g, r = cv2.split(coloredImg)
        RGBimg[File_idx, 0, :, :] = r
        RGBimg[File_idx, 1, :, :] = g
        RGBimg[File_idx, 2, :, :] = b

        if File_idx % 50 == 0:
            print('\nCurrent batch:', str(File_idx))

    save_File_idx = File_idx+1
    for File_idx, imgFile in enumerate(NG_Crack_data_names):
        num_File_idx = save_File_idx+File_idx
        # print("File_idx: ", File_idx)
        coloredImg = cv2.imread(imgFile)

        # IMG Downsampling
        coloredImg = cv2.pyrDown(coloredImg)
        for a in range(3):
            coloredImg = cv2.pyrDown(coloredImg)

        # IMG split
        b, g, r = cv2.split(coloredImg)
        RGBimg[num_File_idx, 0, :, :] = r
        RGBimg[num_File_idx, 1, :, :] = g
        RGBimg[num_File_idx, 2, :, :] = b

        if num_File_idx % 50 == 0:
            print('\nCurrent batch:', str(num_File_idx))

    # s = np.arange(labelimg.shape[0])
    # np.random.shuffle(s)
    # RGBimg = RGBimg[s, :, :, :]
    # labelimg = labelimg[s]
    RGBimg = torch.Tensor(RGBimg)
    labelimg = torch.Tensor(labelimg)

    print("mk dataset")
    detaset = list(range(78))
    batch_size = 5
    for a in range(0,batch_size*78-1,batch_size):
        print(int(a/batch_size))
        detaset[int(a/batch_size)]=(RGBimg[a:a+batch_size,:,:,:],labelimg[a:a+batch_size])

    for batch_idx, (inputs, targets) in enumerate(detaset):
        print(batch_idx, "\n")
        print(inputs.shape, targets.shape)
        print(type(targets))
        print(targets)
