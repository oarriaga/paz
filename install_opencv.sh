mkdir opencv
cd opencv

wget https://github.com/opencv/opencv/archive/4.2.0.zip
unzip 4.2.0.zip

cd opencv-4.2.0
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j8
sudo make install

cd ../../../
rm -rf opencv/