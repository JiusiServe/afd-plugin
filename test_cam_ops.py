# -*- coding: utf-8 -*-
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See LICENSE for more details.
#
# Copyright (C) 2024-2025 Huawei Inc
#
# Author b60078336
# pylint:

"""
：Preparation 
    已部署dispatch send等异步算子，910C环境正常运行

：TestStep
    1. 参数如用例名中参数配置
    2. 按步骤1中的参数调用dispatch send接口，使用用例标注模式，循环1W次，有预期结果1
:AfterTest
    1、清理环境
：ExceptOutput
    1. 1W次算子运算，校验均通过，平均相对误差不超过0.05
"""

import copy
import os
import random
import math
import torch
import torch_npu
import cam
import cam_graph
import logging
import torchair
import torchair as tng

import torch.distributed as dist

from torchair.configs.compiler_config import CompilerConfig

host1_mgt_ip = "172.20.149.65"
ascend_rt_visible_devices = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# dev_id_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
dev_id_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
inner_cycle_cnt = 1
outer_cycle_cnt = 50
verify_result_cnt = 10
weight_cnt = 2
data_type_list = [torch.float16, torch.bfloat16]
quant_mode_list = [1]
MULIT_STREAM = False
GMM_Check = False
IS_graph = False
group_name = ""



class ATTN_Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input_dict, random_nums_attn, comm_args, comm_id,  hidden_size, topk, moe_rank_num, attn_rank_num, expert_num_per_rank, rank, world_size, tp_size):
        
        com_out_list = []
        
        for i in range(inner_cycle_cnt):
            if GMM_Check == True:
                recv_tensor = input_dict[random_nums_attn[i]]["x_out_dtype"]
            else:
                recv_tensor = input_dict[random_nums_attn[i]]["expand_x_out_dtype"]
            com_out=None

            torch.ops.umdk_cam_op_lib.async_dispatch_send(
                input_dict[random_nums_attn[i]]["x_tensor"], input_dict[random_nums_attn[i]]["expert_ids_tensor"], comm_args,comm_id,
                32768,
                input_dict[random_nums_attn[i]]["batch_size"],
                hidden_size,
                topk,
                moe_rank_num,
                attn_rank_num,
                expert_num_per_rank,
                rank,world_size,input_dict[random_nums_attn[i]]["layer_idx"], tp_size, input_dict[random_nums_attn[i]]["quant_mode"],group_name)

            com_out = torch.ops.umdk_cam_op_lib.async_combine_recv(
                recv_tensor, input_dict[random_nums_attn[i]]["expert_ids_tensor"], input_dict[random_nums_attn[i]]["expert_scales"], 
                comm_args,comm_id,
                input_dict[random_nums_attn[i]]["batch_size"],
                hidden_size,
                topk,
                moe_rank_num,
                attn_rank_num,
                expert_num_per_rank,
                rank,world_size,group_name)
            com_out_list.append(com_out)
        return com_out_list
    

class Moe_Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        if IS_graph == True:
            torch_npu.npu.set_compile_mode(jit_compile=False)
            config = torchair.CompilerConfig()
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            if MULIT_STREAM:
                self.compile_forward_multi_stream=torch.compile(self.forward_multi_stream, backend=npu_backend, dynamic=False)
            else:
                self.compile_forward_single_stream=torch.compile(self.forward_single_stream, backend=npu_backend, dynamic=False)
                
    def forward_test(self,input_dict, random_nums_attn, weight_ptr_dict, comm_args, comm_id, bs_max, hidden_size, topk, moe_rank_num, attn_rank_num, expert_num_per_rank, rank, world_size, tp_size):
        
        for i in range(inner_cycle_cnt*(attn_rank_num//tp_size)):
            recv_tensor = input_dict[random_nums_attn[i//(attn_rank_num//tp_size)]]["expand_x_out_dtype"]
            quant_mode = input_dict[random_nums_attn[i//(attn_rank_num//tp_size)]]["quant_mode"]

            dis_out=torch.ops.umdk_cam_op_lib.async_dispatch_recv(
                    recv_tensor, comm_args, comm_id,
                    32768,
                    hidden_size,
                    topk,
                    moe_rank_num,
                    attn_rank_num,
                    expert_num_per_rank,
                    rank,world_size,tp_size,
                    quant_mode,group_name
                )
        
            (expandXOut, expandXOut_shared, dynamicScalesOut, dynamicScalesOut_shared, TokenNums_Rankid_Layeridx, Expert_tokens, Expert_tokens_shared)=dis_out
            if GMM_Check == False:
                TOTAL_Tokens = TokenNums_Rankid_Layeridx[0].item()
                expandXOut_shared_Tokens = 31457
                if quant_mode == 1:
                    expandXOut=expandXOut.to(recv_tensor)
                    expandXOut[:TOTAL_Tokens]=expandXOut[:TOTAL_Tokens]*(dynamicScalesOut[:TOTAL_Tokens].unsqueeze(-1))
                    expandXOut=expandXOut.to(recv_tensor)

                    expandXOut_shared=expandXOut_shared.to(recv_tensor)
                    expandXOut_shared[:expandXOut_shared_Tokens]=expandXOut_shared[:expandXOut_shared_Tokens]*(dynamicScalesOut[:expandXOut_shared_Tokens].unsqueeze(-1))
                    expandXOut_shared=expandXOut_shared.to(recv_tensor)
                result=expandXOut
                result_shared=expandXOut_shared
            else:
                if quant_mode == 0:
                    expandXOut, dynamicScalesOut = torch_npu.npu_dynamic_quant(expandXOut)
                gmGroupList = Expert_tokens
                result = torch.ops.umdk_cam_op_lib.gmm_deq_swiglu_quant_gmm_deq_multi_layer(
                            expandXOut, 
                            weight_ptr_dict["weight1_ptr_tensor"],
                            weight_ptr_dict["scale1_ptr_tensor"],
                            dynamicScalesOut,
                            gmGroupList, 
                            weight_ptr_dict["weight2_ptr_tensor"], 
                            weight_ptr_dict["scale2_ptr_tensor"],
                            TokenNums_Rankid_Layeridx
                        )
            if rank == 8:
                log.info(f"TokenNums_Rankid_Layeridx: {TokenNums_Rankid_Layeridx}")
                log.info(f"Expert_tokens: {Expert_tokens}")

            torch.ops.umdk_cam_op_lib.async_combine_send(
                    result,
                    result_shared,
                    comm_args, 
                    TokenNums_Rankid_Layeridx,
                    comm_id,
                    bs_max,
                    hidden_size,
                    topk,
                    moe_rank_num,
                    attn_rank_num,
                    expert_num_per_rank,
                    rank,
                    world_size,tp_size,group_name
                )

    def forward_single_stream(self,input_dict, random_nums_attn, weight_ptr_dict, comm_args, comm_id, bs_max, hidden_size, topk, moe_rank_num, attn_rank_num, expert_num_per_rank, rank, world_size, tp_size):
        bufLen=1
        buffer=[0]*bufLen
        for i in range(inner_cycle_cnt*(attn_rank_num//tp_size)):
            if i >=bufLen:
                tng.scope.npu_wait_tensor(comm_args,buffer[(i-bufLen)%bufLen])
            recv_tensor = input_dict[random_nums_attn[i//(attn_rank_num//tp_size)]]["expand_x_out_dtype"]
            quant_mode = input_dict[random_nums_attn[i//(attn_rank_num//tp_size)]]["quant_mode"]
            dis_out=torch.ops.umdk_cam_op_lib.async_dispatch_recv(
                    recv_tensor, comm_args, comm_id,
                    bs_max,
                    hidden_size,
                    topk,
                    moe_rank_num,
                    attn_rank_num,
                    expert_num_per_rank,
                    rank,world_size,tp_size,
                    quant_mode,group_name
                )
        
            (expandXOut, expandXOut_shared, dynamicScalesOut, dynamicScalesOut_shared, TokenNums_Rankid_Layeridx, Expert_tokens, Expert_tokens_shared)=dis_out
            TOTAL_Tokens = TokenNums_Rankid_Layeridx[0].item()            
            if GMM_Check == False:
                if quant_mode == 1:
                    expandXOut = torch.as_tensor(expandXOut, dtype=recv_tensor.dtype)
                    expandXOut[:TOTAL_Tokens]=expandXOut[:TOTAL_Tokens]*(dynamicScalesOut[:TOTAL_Tokens].unsqueeze(-1))
                    expandXOut = torch.as_tensor(expandXOut, dtype=recv_tensor.dtype)
                result=expandXOut

            else:
                if quant_mode == 0:
                    expandXOut, dynamicScalesOut = torch_npu.npu_dynamic_quant(expandXOut)
                gmGroupList = Expert_tokens

                result = torch.ops.umdk_cam_op_lib.gmm_deq_swiglu_quant_gmm_deq_multi_layer(
                           expandXOut, 
                            weight_ptr_dict["weight1_ptr_tensor"],
                            weight_ptr_dict["scale1_ptr_tensor"],
                            dynamicScalesOut,
                            gmGroupList, 
                            weight_ptr_dict["weight2_ptr_tensor"], 
                            weight_ptr_dict["scale2_ptr_tensor"],
                            TokenNums_Rankid_Layeridx
                        )
            com_out=torch.ops.umdk_cam_op_lib.async_combine_send(
                    result,
                    comm_args, 
                    TokenNums_Rankid_Layeridx,
                    comm_id,
                    bs_max,
                    hidden_size,
                    topk,
                    moe_rank_num,
                    attn_rank_num,
                    expert_num_per_rank,
                    rank,
                    world_size,tp_size,group_name
                )

            buffer[i%bufLen]=com_out
        return buffer
    
    def forward_multi_stream(self,input_dict, random_nums_attn, weight_ptr_dict, comm_args, comm_id, bs_max, hidden_size, topk, moe_rank_num, attn_rank_num, expert_num_per_rank, rank, world_size, tp_size):
        DISPATCH_AIC_PART = 24
        COMBINE_AIC_PART = 0
    
        DISPATCH_AIV_PART = 24
        COMBINE_AIV_PART = 24

        bufLen=4
        buffer=[torch.tensor([0],device='npu')]*bufLen
        dis_out_buff=torch.tensor([0],device='npu')
        

        for i in range(inner_cycle_cnt*(attn_rank_num//tp_size)):
            recv_tensor = input_dict[random_nums_attn[i//(attn_rank_num//tp_size)]]["expand_x_out_dtype"]
            quant_mode = input_dict[random_nums_attn[i//(attn_rank_num//tp_size)]]["quant_mode"]
            with tng.scope.npu_stream_switch("dispatch_recv_and_gmm_calculate"):
                with tng.scope.limit_core_num(DISPATCH_AIC_PART, DISPATCH_AIV_PART):
                    if i >=bufLen:
                        tng.scope.npu_wait_tensor(comm_args,buffer[(i-bufLen)%bufLen])
                    if i >=1:
                        tng.scope.npu_wait_tensor(comm_args,dis_out_buff)
                    dis_out=torch.ops.umdk_cam_op_lib.async_dispatch_recv(
                        recv_tensor, comm_args, comm_id,
                        bs_max,
                        hidden_size,
                        topk,
                        moe_rank_num,
                        attn_rank_num,
                        expert_num_per_rank,
                        rank,world_size,tp_size,
                        quant_mode,group_name
                    )

                    dis_out_buff=dis_out[0]
                    (expandXOut, expandXOut_shared, dynamicScalesOut, dynamicScalesOut_shared, TokenNums_Rankid_Layeridx, Expert_tokens, Expert_tokens_shared)=dis_out
                    

                    TOTAL_Tokens = TokenNums_Rankid_Layeridx[0].item()                                                                                           
                    if GMM_Check == False:
                        if quant_mode == 1:
                            expandXOut=torch.as_tensor(expandXOut, dtype=recv_tensor.dtype)
                            expandXOut[:TOTAL_Tokens]=expandXOut[:TOTAL_Tokens]*(dynamicScalesOut[:TOTAL_Tokens].unsqueeze(-1))
                            expandXOut=torch.as_tensor(expandXOut, dtype=recv_tensor.dtype)
                        result=expandXOut

                    else:
                        if quant_mode == 0:
                            expandXOut, dynamicScalesOut = torch_npu.npu_dynamic_quant(expandXOut)
                        gmGroupList = Expert_tokens
                        result = torch.ops.umdk_cam_op_lib.gmm_deq_swiglu_quant_gmm_deq_multi_layer(
                                    expandXOut, 
                                    weight_ptr_dict["weight1_ptr_tensor"],
                                    weight_ptr_dict["scale1_ptr_tensor"],
                                    dynamicScalesOut,
                                    gmGroupList, 
                                    weight_ptr_dict["weight2_ptr_tensor"], 
                                    weight_ptr_dict["scale2_ptr_tensor"],
                                    TokenNums_Rankid_Layeridx
                                )

            with tng.scope.npu_stream_switch("combine_send"):
                with tng.scope.limit_core_num(COMBINE_AIC_PART, COMBINE_AIV_PART):
                    com_out=torch.ops.umdk_cam_op_lib.async_combine_send(
                        result,
                        comm_args, 
                        TokenNums_Rankid_Layeridx,
                        comm_id,
                        bs_max,
                        hidden_size,
                        topk,
                        moe_rank_num,
                        attn_rank_num,
                        expert_num_per_rank,
                        rank,
                        world_size,tp_size,group_name
                    )
                    buffer[i%bufLen]=com_out

        return buffer
    
    def forward(self, input_dict, random_nums_attn, weight_ptr_dict, comm_args, comm_id, bs_max, hidden_size, topk, moe_rank_num, attn_rank_num, expert_num_per_rank, rank, world_size, tp_size):
        if IS_graph == True:
            if MULIT_STREAM:
                self.compile_forward_multi_stream(input_dict, random_nums_attn, weight_ptr_dict, comm_args, comm_id, bs_max, hidden_size, topk, moe_rank_num, attn_rank_num, expert_num_per_rank, rank, world_size, tp_size)
            else:
                self.compile_forward_single_stream(input_dict, random_nums_attn, weight_ptr_dict, comm_args, comm_id, bs_max, hidden_size, topk, moe_rank_num, attn_rank_num, expert_num_per_rank, rank, world_size, tp_size)
        else:
            self.forward_test(input_dict, random_nums_attn, weight_ptr_dict, comm_args, comm_id, bs_max, hidden_size, topk, moe_rank_num, attn_rank_num, expert_num_per_rank, rank, world_size, tp_size)

class RunModel(object):
    batch_size = 8
    batch_patch_size = 128
    remainder = 0
    multiple = 0
    batchsize_min= 2070
    batchsize_max= 2070
    hidden_size = 6 * 1024
    down_size = 2048
    up_size = 4096
    topk = 8

    expert_rank_size = 8
    attention_rank_size = 8
    expert_per_rank = 32 # 每卡上路由专家
    tp_size = 8

    share_expert_per_rank = 1 # 每卡上共享专家
    comm_args = None
    comm_id = 0
    x_tensor_list=[]
    x_tensor_quant_list = []
    x_tensor_quant_scale_list = []
    expert_ids_list = []
    expert_scales_list = []

    weight_dict = {}
    input_dict = {} 
    output_dict = {}
    weight_ptr_dict = {}

    eps = 1e-8
    threshold = 0.05
    eager_model = None

    def __init__(self, l_rank_id, g_rank_id, l_rank_size, g_rank_size):
        self.g_rank_id = g_rank_id
        self.l_rank_id = l_rank_id
        self.g_rank_size = g_rank_size
        self.l_rank_size = l_rank_size
        self.g_dev_id = dev_id_list[self.g_rank_id]
        self.l_dev_id = self.g_dev_id % 16

        self.attention_rank_size = self.g_rank_size - self.expert_rank_size  # A rank数量

        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ascend_rt_visible_devices
        os.environ["LCCL_BUFFER_SIZE"] = '8192'
        os.environ["HCCL_INTER_HCCS_DISABLE"] = "FALSE"
        os.environ["LCAL_COMM_ID"] =f"{host1_mgt_ip}:27005"
        self.device = torch.device("npu", self.l_dev_id)
        torch.npu.set_device(self.device)
        torch.npu.config.allow_internal_format = True

        self.month = 5
        self.day = 5
        random.seed(self.day)
        torch.manual_seed(self.day)  # 设置cpu侧随机种子
        torch.npu.manual_seed(100 * self.month + self.day)  # 设置npu侧随机种子

    
    def print_rank_info(self):
        rank_info = {
            "l_rank_id": self.l_rank_id,
            "g_rank_id": self.g_rank_id,
            "l_rank_size": self.l_rank_size,
            "g_rank_size": self.g_rank_size,
            "l_dev_id": self.l_dev_id,
            "g_dev_id": self.g_dev_id
        }

        return rank_info

    def generate_batchsize_random(self, num):
        #num为-1时，batch_size取随机值
        if (num == -1):
            random_batchsize_same = random.randint(self.batchsize_min, self.batchsize_max)
            self.batch_size = random_batchsize_same
        #num为其他整数，batch_size取num
        else:
            self.batch_size = num
        return self.batch_size

    def generate_expand_x_random_tensor(self, data_type, min, max):
        self.remainder = self.batch_size % self.batch_patch_size
        self.multiple = self.batch_size // self.batch_patch_size
        random_tensor = torch.rand((self.batch_patch_size, self.hidden_size), dtype=data_type, device=f"npu:{self.l_dev_id}")
        repeated_tensor = random_tensor.repeat(self.multiple, 1)
        if self.remainder > 0:
            remainder_tensor = random_tensor[:self.remainder]
            # 拼接
            final_tensor = torch.cat([repeated_tensor, remainder_tensor], dim=0)
        else:
            final_tensor = repeated_tensor
        expand_x_random_tensor = final_tensor * max - (final_tensor * min - min)
        return expand_x_random_tensor

    def generate_expert_ids_random_tensor(self):
        # 同一个token发送给moe专家的索引不能重复
        tensor_list = []
        for i in range(self.batch_patch_size):
            tensor_list.append(torch.randperm(self.expert_rank_size * self.expert_per_rank, dtype=torch.int32, device=f"npu:{self.l_dev_id}")[0 : self.topk])
        random_tensor = torch.stack(tensor_list)
        repeated_tensor = random_tensor.repeat(self.multiple, 1)
        if self.remainder > 0:
            remainder_tensor = random_tensor[:self.remainder]
            # 拼接
            expert_ids_random_tensor = torch.cat([repeated_tensor, remainder_tensor], dim=0)
        else:
            expert_ids_random_tensor = repeated_tensor
        return expert_ids_random_tensor

    def generate_scales_random_tensor(self):
        tensor_list = []
        for i in range(self.batch_patch_size):
            scale_tensor = torch.rand((self.topk, ), dtype=torch.float32, device=f"npu:{self.l_dev_id}")
            scale_tensor = scale_tensor / scale_tensor.sum()
            tensor_list.append(scale_tensor)
        random_tensor = torch.stack(tensor_list)
        repeated_tensor = random_tensor.repeat(self.multiple, 1)
        if self.remainder > 0:
            remainder_tensor = random_tensor[:self.remainder]
            # 拼接
            scales_random_tensor = torch.cat([repeated_tensor, remainder_tensor], dim=0)
        else:
            scales_random_tensor = repeated_tensor
        return scales_random_tensor

    def generate_gmm_weight_random_tensor_list(self):
        tensor_list = []
        ptr_list = []
        for i in range(self.expert_rank_size):
            w1_random_tensor= torch.randint(low=-127,high=128, size=(self.expert_per_rank + self.share_expert_per_rank, self.hidden_size, self.up_size), dtype=torch.int8,device="npu").contiguous()
            s1_random_tensor= (torch.rand(size=(self.expert_per_rank + self.share_expert_per_rank, self.up_size), dtype=torch.float32,device="npu")).contiguous()
            w2_random_tensor= torch.randint(low=-127,high=128, size=(self.expert_per_rank + self.share_expert_per_rank, self.down_size, self.hidden_size), dtype=torch.int8,device="npu").contiguous()
            s2_random_tensor= (torch.rand(size=(self.expert_per_rank + self.share_expert_per_rank, self.hidden_size), dtype=torch.float32,device="npu")).contiguous()
            weight_dict = { "weight1":w1_random_tensor,
                            "scale1":s1_random_tensor,
                            "weight2":w2_random_tensor,
                            "scale2":s2_random_tensor}
            weight_ptr_dict = {
                "weight1_ptr":w1_random_tensor.data_ptr(),
                "scale1_ptr":s1_random_tensor.data_ptr(),
                "weight2_ptr":w2_random_tensor.data_ptr(),
                "scale2_ptr":s2_random_tensor.data_ptr()
            }
            tensor_list.append(weight_dict)
            ptr_list.append(weight_ptr_dict)
        return tensor_list, ptr_list

    def create_cam_comm(self):
        # self.comm_args = cam.create_comm_moe(self.comm_id, self.g_rank_id, self.g_rank_size, self.batchsize_max, self.hidden_size, self.topk, self.expert_rank_size, f"{host1_mgt_ip}:27007", True).to("npu")
        return
        
    def enable_model(self):
        if self.g_rank_id < self.attention_rank_size:
            self.eager_model = ATTN_Module().npu()
        else:
            self.eager_model = Moe_Model().npu()

    def exec_eager_model(self,random_nums_attn):
        if self.g_rank_id < self.attention_rank_size:
            self.output_list = self.eager_model(self.input_dict, random_nums_attn, self.comm_args, self.comm_id,  
            self.hidden_size, self.topk, self.expert_rank_size, self.attention_rank_size, self.expert_per_rank, self.g_rank_id, self.g_rank_size,self.tp_size)
        else:
            self.eager_model(self.input_dict, random_nums_attn, self.weight_ptr_dict, self.comm_args, self.comm_id,  
            self.batchsize_max,self.hidden_size, self.topk, self.expert_rank_size, self.attention_rank_size, self.expert_per_rank, self.g_rank_id, self.g_rank_size,self.tp_size)

    def moe_expert_multiple_caculate(self, expandX, expertIds, topk_weight, expend_list, expand_idx, expert_tokens, scale_list, expert_buffers, remainder_buffers, Topk, expert_start, expert_end):
        patch_expandX = expandX[:self.batch_patch_size]
        patch_expertIds = expertIds[:self.batch_patch_size]
        patch_topk_weight = topk_weight[:self.batch_patch_size]
        for token_idx in range(self.batch_patch_size):
            for k in range(Topk):
                expert_id = patch_expertIds[token_idx][k]
                if expert_start <= expert_id < expert_end:
                    rel_expert_idx = expert_id - expert_start
                    weight = patch_topk_weight[token_idx][k] if patch_topk_weight is not None else None
                    x = patch_expandX[token_idx]
                    expert_buffers[rel_expert_idx].append((token_idx, x, weight))
                    if token_idx < self.remainder:
                        remainder_buffers[rel_expert_idx] += 1
    
    # 按专家顺序处理收集到的token
        for rel_expert_idx in range(self.expert_per_rank):
            buffer = expert_buffers[rel_expert_idx]
            if not buffer:
                # 如果没有 token，该专家输出应为 0 或保持默认，视具体逻辑而定
                # 这里假设 token_count 为 0
                expert_tokens.append(0) 
                continue
            token_count = len(buffer)*self.multiple + remainder_buffers[rel_expert_idx]
            expert_tokens.append(token_count)
            list_add = []
            expand_idx_add = []
            scale_list_add = []
            for token_idx, x, weight in buffer:
                list_add.append(x)
                expand_idx_add.append(token_idx)    
                if patch_topk_weight is not None:
                    scale_list_add.append(weight)
            if list_add:
                list_add = list_add * self.multiple + list_add[:remainder_buffers[rel_expert_idx]]
                scale_list_add = scale_list_add * self.multiple + scale_list_add[:remainder_buffers[rel_expert_idx]]
                
                # 保存原始索引的副本
                expand_idx_add_orgin = copy.deepcopy(expand_idx_add)
                expand_idx_add_remainder = copy.deepcopy(expand_idx_add[:remainder_buffers[rel_expert_idx]])
                expand_idx_add_remainder = [val + self.multiple*self.batch_patch_size for val in expand_idx_add_remainder]
                for i in range(1, self.multiple):
                    expand_idx_add.extend([val + (i)*self.batch_patch_size for val in expand_idx_add_orgin])
                expand_idx_add = expand_idx_add + expand_idx_add_remainder
                expend_list.extend(list_add)
                expand_idx.extend(expand_idx_add)
                scale_list.extend(scale_list_add)
        return expend_list, expand_idx, scale_list

    def moe_expert_caculate(self, remainder_expandX, remainder_expertIds, remainder_topk_weight, expend_list, expand_idx, expert_tokens, scale_list, expert_buffers, Topk, expert_start, expert_end):
        for token_idx in range(self.remainder):
            for k in range(Topk):
                expert_id = remainder_expertIds[token_idx][k]
                if expert_start <= expert_id < expert_end:
                    rel_expert_idx = expert_id - expert_start
                    weight = remainder_topk_weight[token_idx][k] if remainder_topk_weight is not None else None
                    x = remainder_expandX[token_idx]
                    expert_buffers[rel_expert_idx].append((token_idx, x, weight))
        
        # 按专家顺序处理收集到的token
        for rel_expert_idx in range(self.expert_per_rank):
            buffer = expert_buffers[rel_expert_idx]
            if not buffer:
                # 如果没有 token，该专家输出应为 0 或保持默认，视具体逻辑而定
                # 这里假设 token_count 为 0
                expert_tokens.append(0) 
                continue
            token_count = len(buffer)
            expert_tokens.append(token_count)
            list_add = []
            expand_idx_add = []
            scale_list_add = []
            for token_idx, x, weight in buffer:
                list_add.append(x)
                expand_idx_add.append(token_idx)
                if remainder_topk_weight is not None:
                    scale_list_add.append(weight)
            if list_add:
                expend_list.extend(list_add)
                expand_idx.extend(expand_idx_add)
                scale_list.extend(scale_list_add)
        return expend_list, expand_idx, scale_list
    
    def dispatch_simulate(self, expandX, expertIds, moe_idx, topk_weight=None):
        TokenNums, Topk = expertIds.shape
        expend_list, expert_tokens, expand_idx,scale_list = [],[],[],[]
        
        MOE_RANK = moe_idx - self.attention_rank_size
        # 计算共享token分配
        shared_tokens = [TokenNums // self.expert_rank_size] * self.expert_rank_size
        for i in range(TokenNums % self.expert_rank_size):
            shared_tokens[i] += 1
        for i in range(1, self.expert_rank_size):
            shared_tokens[i] += shared_tokens[i-1]
        
        start = 0 if MOE_RANK == 0 else shared_tokens[MOE_RANK-1]
        end = shared_tokens[MOE_RANK]
        
        # 处理共享token
        expand_idx.extend(range(start, end))
        expend_list.extend(expandX[start:end])
        if topk_weight is not None:
            scale_list.extend([1.0] * (end - start))
        shared_count = end - start
        expert_tokens.append(shared_count)
        
        # 预计算专家范围
        expert_start = self.expert_per_rank * MOE_RANK
        expert_end = expert_start + self.expert_per_rank
        
        # 为每个专家创建缓冲区
        expert_buffers = [[] for _ in range(self.expert_per_rank)]
        remainder_buffers= [0 for _ in range(self.expert_per_rank)]

        remainder_expandX = expandX[:self.remainder]
        remainder_expertIds = expertIds[:self.remainder]
        remainder_topk_weight = topk_weight[:self.remainder]
        # 单次遍历收集所有专家分配
        if self.multiple > 0:
            expend_list, expand_idx, scale_list = self.moe_expert_multiple_caculate(expandX, expertIds, topk_weight, expend_list, expand_idx, expert_tokens, scale_list, expert_buffers, remainder_buffers, Topk, expert_start, expert_end)
        else:
           expend_list, expand_idx, scale_list = self.moe_expert_caculate(remainder_expandX, remainder_expertIds, remainder_topk_weight, expend_list, expand_idx, expert_tokens, scale_list, expert_buffers, Topk, expert_start, expert_end)
        
        # 构造结果张量
        if len(expend_list) == 0:
            hidden_states = torch.empty((0, self.hidden_size), dtype=expandX.dtype, device=self.device)
        else:
            hidden_states = torch.stack(expend_list, dim=0).to(self.device)
        expert_tokens = torch.tensor(expert_tokens, dtype=torch.int64, device=self.device)
        return hidden_states, expert_tokens, expand_idx, scale_list

  
    def prefill_simulate(self, expand_x,expert_ids,topk_weight, expert_weight_list, data_type):
        topk_weight=topk_weight.to(torch.float)
        result = torch.zeros(expand_x.shape,dtype=torch.float32,device="npu")
        expand_x=expand_x.cpu()
        expert_ids=expert_ids.cpu()
        for i in range(self.expert_rank_size):
            hidden_states,expert_tokens,expand_idx,scale_list=self.dispatch_simulate(expand_x,expert_ids,i+self.attention_rank_size,topk_weight)
            gmPermuteWeightOne,gmPermuteScaleOne,gmWeightTwo,gmScaleTwo=expert_weight_list[i]["weight1"], expert_weight_list[i]["scale1"], expert_weight_list[i]["weight2"], expert_weight_list[i]["scale2"]

            hidden_states_int8, dynamicScalesOut = torch_npu.npu_dynamic_quant(hidden_states)
            res=torch.ops.umdk_cam_op_lib.gmm_deq_swiglu_quant_gmm_deq(
                    hidden_states_int8, 
                    gmPermuteWeightOne,
                    gmPermuteScaleOne,
                    dynamicScalesOut,
                    expert_tokens, 
                    gmWeightTwo, 
                    gmScaleTwo
                )
            res=res.to(torch.float32)
            for j in range(len(expand_idx)):
                result[expand_idx[j]]=result[expand_idx[j]]+res[j]*scale_list[j]
        return result.to(torch.bfloat16)

    def generate_combine_verify(self, expandX, expertIds, scales,expert_weight, data_type):
        Ans=self.prefill_simulate(expandX, expertIds, scales, expert_weight, data_type)
        return Ans
    
    def dispatch_combine_verify(self, random_nums_attn):
        
        if GMM_Check == True:
            for i in range(len(random_nums_attn)):
                expect = (self.output_dict[random_nums_attn[i]]["combine_simulate"])
                absolute_tolerance = self.threshold * torch.mean(torch.abs(expect)).item()
                expect_safe = expect + torch.where(expect == 0, torch.tensor(absolute_tolerance, device=expect.device), torch.tensor(0, device=expect.device))
                act = self.output_list[i]
                relative_error = torch.abs(expect-act)/torch.abs(expect_safe)
                mean_relative_error = torch.mean(relative_error).item()
                if mean_relative_error >= self.threshold or math.isnan(mean_relative_error):
                    raise ValueError(f'[ERROR]comm_quant_mode verify_combine_output failed !!! expect{expect}act+++++{act}Found mean_relative_error exceeding tolerance!{mean_relative_error}') 
        else:
            for i in range(len(random_nums_attn)):
                expect = (self.input_dict[random_nums_attn[i]]["x_tensor"]*2)
                act = self.output_list[i]
                expect_safe = expect + torch.where(expect == 0, torch.tensor(self.eps, device=expect.device), torch.tensor(0, device=expect.device))
                mean_relative_error = torch.mean(torch.abs(expect-act)/torch.abs(expect_safe)).item()
                if mean_relative_error >= self.threshold or math.isnan(mean_relative_error):
                    log.info(f"self.g_rank_id{self.g_rank_id}mean_relative_error{mean_relative_error}")
                    log.info(f"expect x = {expect}, expect x shape= {expect.shape}")
                    log.info(f"actral x = {act}, actral x shape= {act.shape}")
                    raise ValueError(f'[ERROR]comm_quant_mode verify_combine_output failed !!!')
    
        return mean_relative_error

    def run_test(self):
        # 生成expand_x随机tensor
        quant_mode = random.choice(quant_mode_list)
        data_type = random.choice(data_type_list)
        random_weight_index = random.choices(range(weight_cnt), k=verify_result_cnt)
        expand_x_out_dtype = torch.tensor([0], dtype=data_type, device='npu')
        x_out_dtype = torch.tensor([0], dtype=torch.bfloat16, device='npu')
        
        weight1_ptr_list, scale1_ptr_list, weight2_pt_list, scale2_ptr_list = [],[],[],[]
        final_weight1_ptr_list, final_scale1_ptr_list, final_weight2_pt_list, final_scale2_ptr_list = [],[],[],[]
        
        for i in range(weight_cnt):
            expert_weight_list, weight_ptr_list= self.generate_gmm_weight_random_tensor_list()
            self.weight_dict[i] = {"expert_weight_list":expert_weight_list, "weight_ptr_list":weight_ptr_list}
        for i in range(weight_cnt):
            weight_ptr_list = self.weight_dict[i]["weight_ptr_list"]
            if self.g_rank_id >= self.attention_rank_size:
            # 提取指定 rank_id 的 w1_random_tensor 的 data_ptr
                weight1_ptr = weight_ptr_list[self.g_rank_id-self.attention_rank_size]["weight1_ptr"]
                scale1_ptr = weight_ptr_list[self.g_rank_id-self.attention_rank_size]["scale1_ptr"]
                weight2_ptr = weight_ptr_list[self.g_rank_id-self.attention_rank_size]["weight2_ptr"]
                scale2_ptr = weight_ptr_list[self.g_rank_id-self.attention_rank_size]["scale2_ptr"]
                weight1_ptr_list.append(weight1_ptr)
                scale1_ptr_list.append(scale1_ptr)
                weight2_pt_list.append(weight2_ptr)
                scale2_ptr_list.append(scale2_ptr)
        if self.g_rank_id >= self.attention_rank_size:
            for i in range(verify_result_cnt):
                final_weight1_ptr_list.append(weight1_ptr_list[random_weight_index[i]])
                final_scale1_ptr_list.append(scale1_ptr_list[random_weight_index[i]])
                final_weight2_pt_list.append(weight2_pt_list[random_weight_index[i]])
                final_scale2_ptr_list.append(scale2_ptr_list[random_weight_index[i]])
            

        # # 注意：data_ptr() 返回的是 Python int，可以安全转成 torch.int64
            weight1_ptr_tensor = torch.tensor(final_weight1_ptr_list, dtype=torch.int64,device="npu").contiguous()
            scale1_ptr_tensor = torch.tensor(final_scale1_ptr_list, dtype=torch.int64,device="npu").contiguous()
            weight2_ptr_tensor = torch.tensor(final_weight2_pt_list, dtype=torch.int64,device="npu").contiguous()
            scale2_ptr_tensor = torch.tensor(final_scale2_ptr_list, dtype=torch.int64,device="npu").contiguous()

            self.weight_ptr_dict = {"weight1_ptr_tensor":weight1_ptr_tensor, "scale1_ptr_tensor":scale1_ptr_tensor, "weight2_ptr_tensor":weight2_ptr_tensor, "scale2_ptr_tensor": scale2_ptr_tensor}


        for i in range(verify_result_cnt): 
            batch_size = self.generate_batchsize_random(-1)       
            x_tensor = self.generate_expand_x_random_tensor(data_type, -100, 100)
            expert_ids_tensor = self.generate_expert_ids_random_tensor()
            expert_scales= self.generate_scales_random_tensor()
            expert_weight_list = self.weight_dict[random_weight_index[i]]["expert_weight_list"]
            if self.g_rank_id < self.attention_rank_size:
                combine_simulate = self.generate_combine_verify(x_tensor, expert_ids_tensor, expert_scales, expert_weight_list, data_type)
            else:
                combine_simulate = []
            self.input_dict[i] = {"batch_size":batch_size, "expand_x_out_dtype":expand_x_out_dtype, "x_out_dtype": x_out_dtype,"quant_mode":quant_mode, 
                                "x_tensor":x_tensor, "expert_ids_tensor":expert_ids_tensor, "expert_scales":expert_scales, "layer_idx":i}
            self.output_dict[i] = {"combine_simulate":combine_simulate}
        
        
        
        # 创建cam通信域
        self.create_cam_comm()
        # 启用模式
        self.enable_model()
        #循环outer_cycle_cnt次       
        for i in range(outer_cycle_cnt):
            random.seed(self.day + (self.g_rank_id//self.tp_size) + i)
            random_nums_attn = random.choices(range(verify_result_cnt), k=inner_cycle_cnt)
            if self.g_rank_id == 1:
                log.info(f"{i + 1}th cycle...")
            self.exec_eager_model(random_nums_attn)
            if self.g_rank_id < self.attention_rank_size:
                mean_relative_error = self.dispatch_combine_verify(random_nums_attn)



if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    ranks_list = list(np.arange(0, world_size))
    group = dist.new_group(backend="hccl", ranks=ranks_list)
    group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    actor = RunModel(local_rank, rank, 16, world_size)
    actor.run_test()