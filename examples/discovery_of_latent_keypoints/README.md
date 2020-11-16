This example requires to install two additional libraries for rendering 
```
pip install pyrender --user
pip install trimesh --user
```
If you are running the script on a remote machine via SSH, then might run 
into the following error:

```
ValueError: Failed to initialize Pyglet window with an OpenGL >= 3+ context. If you're logged in via SSH, ensure that you're running your script with vglrun (i.e. VirtualGL). The internal error message was ""
```
To fix this, you need to run the PyOpenGL in a headless configuration which is not enabled by default. Just uncomment the following line(4) in <em>discover_latent_keypoints.py</em>
```
# os.environ["PYOPENGL_PLATFORM"] = 'egl'
```
This will use the GPU accelerated rending on your remote machine. To use CPU-accelerated rendering, your need to use
OSMesa instead of EGL. However, this is not tested yet. 
```
# os.environ["PYOPENGL_PLATFORM"] = 'osmesa'
```

If you run into the following error: 
```
tensorflow.python.framework.errors_impl.UnknownError: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above. [Op:Conv2D]
```
Either you are short on memory (reduce batch size) or uncomment the following line(5) in <em>discover_latent_keypoints.py</em>

```
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
```