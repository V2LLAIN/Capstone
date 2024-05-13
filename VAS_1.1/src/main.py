import time
import VAS_system


if __name__ == "__main__":
    MODE_IP_CAM = 0
    MODE_VIDEO = 1
    VAS = VAS_system.VAS_system()


    ''' korea LP '''
    # Video_path = "/home/kkk/Desktop/Dataset/TEST/110708/"
    # cam_names = ["2020-11-07-21.10.30"]
    # Start_frame = "3900"    # "A": 3900 / "A": 14940

    # Video_path = "/home/kkk/Desktop/Dataset/TEST/1120/"
    # cam_names = ["2020-11-20-12.40.30"]
    # Start_frame = "15000"   

    # Video_path = "/home/kkk/admin/VAS_data/VAS_test_videos/"
    # cam_names = ["2020-11-20-11.40.30"]
    # Start_frame = "6500"  # "FA": 0 / "A": 3200 / "E": 4968 / "C": 6500 / "A": 8300 / "C": 10800 / "C": 12800 / "A": 15300 / "C": 17100

    # Video_path = "/home/kkk/Desktop/Dataset/TEST/110708/"
    # cam_names = ["2020-11-07-18.50.30"]
    # Start_frame = "14400"   

    # ''' English LP Local video testing'''
    # Video_path = "/home/admin/VAS_data/VAS_test_videos/TEST/english_lp_video/"
    # cam_names = ["2020-11-20-11.40.30"]
    # Start_frame = "0"   

    ''' BCT '''
    Video_path = "/home/admin/VAS_data/VAS_test_videos/TEST/BCT_test_video/"
    cam_names = ["record_2022-06-15-10.20.30"]
    Start_frame = "200"       


    for name in cam_names:
        video_start = time.time()

        print("[ Video Idx: ", name, " ]")
        video_last_name = name + ".mp4"
        Video_Info = [Video_path, video_last_name, Start_frame]
        VAS.system_Operation(Video_Info, MODE_VIDEO)

        print(
            "\n\t> Elapsed [ ", name, " ] Video Time: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - video_start)), "(HH:MM:SS)",
        )
