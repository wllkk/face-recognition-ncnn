# face-recognition-ncnn

- for x86_64 linux  with no vulkan

```cmake
mkdir build
cd build
cmake ../
make -j4
```

- for jetson agx  
  交叉编译armv8版本的ncnn， 使用install/lib、install/include 替换本项目中的lib、include.  
  在jetson agx下执行：

```cmake
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/jetson.toolchain.cmake ../
make -j4
```



- References

> https://github.com/Tencent/ncnn
>
> https://github.com/Charrin/RetinaFace-Cpp
>
> https://github.com/deepinsight/insightface

