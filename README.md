# HouseNumberDetection
CNN and Object Detection

1. In linux/mac, setup conda env with provided yml file, and make sure pytorch has properly setup. And activate the conda env


2. run command: python run.py

3. from above command, digits in five images would be labeled and saved to graded_images folder. 

4. if TA needs to re-run training process, download train_32x32.mat, test_32x32.mat, extra_32x32.mat, and train.tar.gz file  from http://ufldl.stanford.edu/housenumbers/. unzip train.tar.gz file into working directory. Then run command: nohup python -u cnn.py > cnn_tuned.out &

5. from above command, training process logs will be recorded into cnn_tuned.out file, and re-trained VGG16 model will be saved as svhn_11_vgg16_tuned.pth, and re-trained my own CNN model will be saved as svhn_11_cnn5_tuned.pth into working directory

6. The output images 1.png to 5.png were also included in submission. 
