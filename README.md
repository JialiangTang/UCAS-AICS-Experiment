# UCAS-AICS-Experiment
中国科学院大学2022秋季学期智能计算系统-陈云霁

含实验代码、必要材料

仅供学习参考，请勿抄袭

exp3-3 池化加速部分借鉴了@LuoXuKun的代码，但个人认为他对张量的处理略显混乱。

exp4-4容易编译失败，可以把BUILD里的job nums修改为更低的16或者8，如果/root/.cache/bazel/_bazel_root处报错，执行rm -rf /root/.cache/bazel/_bazel_root将其删除
