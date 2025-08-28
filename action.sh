sudo docker run --gpus all -it - v /home/labadmin/Documents/mainrun:/workspace mainrun-env
git config --global --add safe.directory /workspace
git config --global user.email "teedsingyau@gmail.com"
git config --global user.name "Ted Lo"
task train