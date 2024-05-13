# Configuration
- On [config.json](/data/config.json)

# Build and Test

* **Install & Activate & Deactivate virtual Env**
    * **[Virtualenv]**
        * (If not Installed -> Install)
            * pip install virtualenv
            * virtualenv (env_name)
            * source (env_name)/bin/activate
            * pip install -r (env_name)/requirements.txt
            <br>
        * (If Installed -> Activate)
            * source (env_name)/bin/activate
            <br>
        * (If Installed -> Deactivate)
            * deactivate
            <br>
            <br>

    * **[Anaconda]**
        * (If not Installed -> Install)
            * goto site(https://www.anaconda.com/download/#linux) and install 
            * source activate (env_name)
            <br>
        * (If Installed -> Activate)
            * source activate (env_name)
            <br>
        * (If Installed -> Deactivate)
     
          #####
    rviz2 -d install/urdf_tutorial_r2d2/share/urdf_tutorial_r2d2/r2d2.rviz
            * conda deactivate
            <br>
<br>
* **execute python script**
    * cd src
    * python [main.py](/src/main.py) - Test using a video
    * python [main_set_test.py](/src/main_set_test.py) - Test for many videos
    * python [main_IPcamera.py](/src/main_IPcamera.py) - Test for IP camera

# Repo

# Applied function
- Container OCR
- ANPR
- Seperate Chassis Position
- Chassis Length
- Back Door Direction
- Image Stitching(Left, Right)
- Seal Presence
- Signal BoomBarrier Open
- Images, Videos Writer
- Send2GOS using RestAPI (Truck arrive, Boom barrier open, Truck information, stitching information)


# Mode control
- System control using each FLAG in [config.json](/data/config.json) (Best to use only **SENDING_FLAG** for speed optimization)

<pre>
<code>
SENDING_FLAG = True
VIEWER_FLAG = True
DEBUG_FLAG = True
LANGUAGE_FLAG = 1 or 2
</code>
</pre>

> **SENDING_FLAG**    
: Send output(json) 2 GOS using Rest-API and write images, videos function control

> **VIEWER_FLAG**   
: Control the system debugging while viewing the result as shown in the picture below.
<img src=".img/result.jpg" width="50%" height="50%" title="VIEWER_FLAG" alt="Viewer"></img>

> **DEBUG_FLAG**   
: Processing time check, Writing result txt function control

> **LANGUAGE_FLAG**   
: Select LP rule according to language (1: Korean, 2: English)
