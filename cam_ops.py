com_out = torch.ops.umdk_cam_op_lib.async_dispatch_send(
    x: Tensor, # 要发送的 token, shape = [tokenNum, hiddenSize], dtype = float16 or bfloat16
    expertIds: Tensor, # 每个 token 的 top-k 路由, shape = [tokenNum, topKNum], dtype = int32
    commArgs: Tensor, # 通信域句柄
    commId: int, # 通信 ID
    maxSeqLen: int, # 序列的最大可能长度
    batchSize: int, # tokenNum
    hiddenSize: int, # hiddenSize
    topk: int, # topKNum
    expertRankSize: int, # MoE Rank 数
    attentionRankSize: int, # Attention Rank 数
    expertPerRank: int, # MoE 每卡专家数
    rank: int, # 当前 Rank 的全局 id (在所有 Attention 和 MoE 卡中的总 id)
    worldSize: int, # Attention + MoE 总卡数
    layerIndex: int, # layer 标识, 从第 0 层开始计数
    tpSize: int, # Attention 侧的 tp size (实际上是 CP size)
    dynamicQuant: int, # 是否在传输中进行动态量化 0: 不量化, 1: 量化
    groupName: str, # HCCL 通信域名字
)
com_out # 执行完成标记位

dis_out = torch.ops.umdk_cam_op_lib.async_dispatch_recv(
    x: Tensor, # shape = [1], dtype = float16 or bfloat16, 占位
    commArgs: Tensor, # 通信域句柄
    commId: int, # 通信 ID
    batchSize: int, # 序列的最大可能长度
    hiddenSize: int, # hiddenSize
    topk: int, # topKNum
    expertRankSize: int, # MoE Rank 数
    attentionRankSize: int, # Attention Rank 数
    expertPerRank: int, # MoE 每卡路由专家数
    rank: int, # 当前 Rank 的全局 id (在所有 Attention 和 MoE 卡中的总 id)
    worldSize: int, # Attention + MoE 总卡数
    layerIndex: int, # layer 标识, 从第 0 层开始计数
    tpSize: int, # Attention 侧的 tp size (实际上是 CP size)
    dynamicQuant: int, # 是否在传输中进行动态量化 0: 不量化, 1: 量化
    groupName: str, # HCCL 通信域名字
)
dis_out = ( # 输出全部都是 Tensor
    expandXOut, # 接收到的路由专家 token, shape = [256k (算子中固定), hiddenSize], dtype = int8 (量化)
    expandXOut_shared, # 接收到的共享专家 token, shape = [256k / ep_size, hiddenSize] ,dtype = int8
    dynamicScalesOut, # 接收到的路由专家的量化 scale, shape = [256k], dtype = float32
    dynamicScalesOut_shared, # 接收到的共享专家的量化 scale, shape = [256k / ep_size], dtype = float32
    TokenNums_Rankid_Layeridx, # token 数等信息, shape = [5 + tpSize * (2 + expertPerRank)], dtype = int64, 具体定义如下
    # [36864,  实际收到的 token 总数
    # 0,       本轮收到的 token 所来自的 DP 组中第一个 Rank 的全局 id
    # 0,       层数
    # 1,       本轮接收的起始专家的本卡 id
    # 32,      本轮接收的终止专家的本卡 id
    # 27648, 27648, 27648, 27648, 27648, 27648, 27648, 27648, 每个 attention cp rank 在发 token 给本卡之前已经给其它卡发了多少 token
    # 512, 128, ..., 128, (32 个 128) 本卡从 attention cp 0 收取了 512 个共享专家 token， 并且 32 个路由专家都收取了 128 个 token
    # 512, 128, ..., 128, (32 个 128) 本卡从 attention cp 1 收取了 512 个共享专家 token， 并且 32 个路由专家都收取了 128 个 token
    # ......
    # 512, 128, ..., 128] (32 个 128) 本卡从 attention cp 8 收取了 512 个共享专家 token， 并且 32 个路由专家都收取了 128 个 token
    Expert_tokens, # 每个路由专家接收到的 token 总数, shape = [expertPerRank], dtype = int64
    Expert_tokens_shared # 共享专家接收到的 token 总数, shape = [1], dtype = int64
)

com_out = torch.ops.umdk_cam_op_lib.async_combine_send(
    expandX: Tensor, # 完成计算后的路由专家 token, shape = [256k (算子中固定), hiddenSize], dtype = float16 or bfloat16
    expandXShared: Tensor, # 完成计算后的共享专家 token, shape = [256k / ep_size, hiddenSize], dtype = float16 or bfloat16
    commArgs: Tensor, # 通信域句柄
    expertTokenNums: Tensor, # token 数等信息, 同 dispatch_recv 的接收结果
    commId: int, # 通信 ID
    batchSize: int, # tokenNum
    hiddenSize: int, # hiddenSize
    topk: int, # topKNum
    expertRankSize: int, # MoE Rank 数
    attentionRankSize: int, # Attention Rank 数
    expertPerRank: int, # MoE 每卡专家数
    rank: int, # 当前 Rank 的全局 id (在所有 Attention 和 MoE 卡中的总 id)
    worldSize: int, # Attention + MoE 总卡数
    tpSize: int, # Attention 侧的 tp size (实际上是 CP size)
    groupName: str, # HCCL 通信域名字
)
com_out # 执行完成标记位

com_out = torch.ops.umdk_cam_op_lib.async_combine_recv(
    expandX: Tensor, # shape = [1], dtype = float16 or bfloat16, 占位
    expertIds: Tensor, # 每个 token 的 top-k 路由, shape = [tokenNum, topKNum], dtype = int32, 同 dispatch_send 的 expertIds
    expertScales: Tensor, # topk_weight 用于加权求和, shape = [tokenNum, topKNum], dtype = float32
    commArgs: Tensor, # 通信域句柄
    commId: int, # 通信 ID
    batchSize: int, # tokenNum
    hiddenSize: int, # hiddenSize
    topk: int, # topKNum
    expertRankSize: int, # MoE Rank 数
    attentionRankSize: int, # Attention Rank 数
    expertPerRank: int, # MoE 每卡专家数
    rank: int, # 当前 Rank 的全局 id (在所有 Attention 和 MoE 卡中的总 id)
    worldSize: int, # Attention + MoE 总卡数
    groupName: str, # HCCL 通信域名字
)
com_out # 收到的所有 token 结果, shape, dtype 与 dispatch_send 的 x 一致