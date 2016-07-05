## Create Instance
- ami-b141a2f5
- Instance type g2.8xLarge
- Add storage: change root storage to larger amount. Currently tried 512 Gb
- Tag with Project = Unblur

- Use the keypair unblur.pem
  - Find this by looking up "unblur keypair"


## Log in
- locally: chmod 700 path_to_unblur.pem
- ssh-add path_to_unbur
- ssh ubuntu@the_ip_address

## Set up environment
- Install conda
  - wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda-2.3.0-Linux-x86_64.sh
  - conda update conda
  - conda update anaconda
  - conda create -n py35 python=3.5 anaconda
  - source activate py35
- Install opencv3
  - `conda install -c https://conda.binstar.org/menpo opencv3` does installation. Linking to ffmpeg is open question
  - In theory, we could do `conda install scikit-video` as possible workaround for opencv3 not being able to use VideoCapture due to ffmpeg prob. Instead, it's easier to build the images in macos where the ffmpeg link isn't an issue, and push the static images to ec2.



- Update theano
  - pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
  - edit ~/.theanorc. Set floatX to float16

- Install Cudnn
  - curl -fsSL http://developer.download.nvidia.com/compute/redist/cudnn/v4/cudnn-7.0-linux-x64-v4.0-prod.tgz -O
  - sudo tar -xzf cudnn-7.0-linux-x64-v4.0-prod.tgz
  - go into the cudnn directory and copy contents into matching directories under  /usr/local/cuda-7.0
  - python -c "import theano" to verify it works
  - rm ~/cudnn-7.0-linux-x64-v4.0-prod.tgz
  - sudo rm -r ~/cuda
  - change floatx in .theanorc to float16

- pip install keras

- using sftp, put any desired data in the /home/ubuntu/unblur/data directory

## Set up Remote Syncing from Atom
 - Instructions [here](https://atom.io/packages/remote-sync)
 - Set up monitoring on appropriate files
