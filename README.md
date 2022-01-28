# GTA_ADAS
Detecting the pedestrian and vehicle in the front. If the object is too close to your car, system will show you a alarm to avoid collision</br>
![Showcase](https://github.com/OpenVINO-dev-contest/GTA_ADAS/blob/main/image/demo.gif)

# How to install
1. Install Python on your PC, and we recommand Python 3.7 for your first try</br>
https://www.python.org/downloads/
2. Start your CMD terminal and run "pip install -r requirement.txt", to install all the dependancy
3. run "python main.py -m ./model/pedestrian-and-vehicle-detector-adas-0001.xml"

# How it works
The program will capture 672X384 from the screen , and then</br>
figure out the position of any person and vehicle in front of you.

# Tips
1. Do not scale your Windows sreen</br>
![Showcase](https://github.com/OpenVINO-dev-contest/GTA_ADAS/blob/main/image/donotscale.png)</br>
2. This application only works on 1920x1080 resolution