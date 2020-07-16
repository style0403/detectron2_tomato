import rosbag
from sensor_msgs import point_cloud2

bag = rosbag.Bag('/bag_zivid/2019-12-12-21-50-28-002.bag')
for topic,msg, t in bag.read_messages(topics=['/zivid_camera/points']):
    print(msg)
bag.close()