### 临时项目
    本项目暂时用来计算模板坐标系下的小球坐标和兴趣点坐标（三个及以上小球）
    （如果只有两个小球就直接去json输一个全0原点和另一个点坐标）
    
### 使用方式
- 在 xxx.json 中填入对应的坐标（比如fusion360中原点坐标系下每个点的坐标）
- 运行 transform_from_some_coordinate.py   
- 把 output_xxx.json 中更改的值粘贴到相机项目的配置文件的对应位置
- 其他.py用于测试