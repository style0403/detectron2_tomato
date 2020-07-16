import zivid
app = zivid.Application()
camera = app.connect_camera()
frame = camera.capture()
frame.save("my-frame.zdf")