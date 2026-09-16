Pin `hydra-core < 1.3.7`: 1.3.7 refuses to instantiate the `hydra._internal` sweeper target that hydra-zen's `launch` names directly, which made every sweep fail with `InstantiationException`.
