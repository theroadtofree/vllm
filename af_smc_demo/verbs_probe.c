/* verbs ABI 探测:一次性打印本机 verbs.h 的真实结构体偏移与 attr_mask
 * 值,用于校准 verbs_demo.py / verbs_wire.py 的 ctypes 定义。
 *
 * 编译运行(仅依赖头文件,无需链接):
 *   gcc -o /tmp/verbs_probe verbs_probe.c && /tmp/verbs_probe
 * 把完整输出发回即可。
 */
#include <stdio.h>
#include <stddef.h>
#include <infiniband/verbs.h>

#define O(T, F) printf("offsetof(" #T "." #F ")=%zu\n", offsetof(T, F))
#define S(T) printf("sizeof(" #T ")=%zu\n", sizeof(T))
#define M(F) printf("mask " #F "=0x%llx\n", (unsigned long long)(F))

int main(void)
{
    printf("== struct sizes ==\n");
    S(struct ibv_qp_attr);
    S(struct ibv_ah_attr);
    S(struct ibv_global_route);
    S(struct ibv_qp_init_attr);
    S(struct ibv_qp_cap);
    S(struct ibv_send_wr);
    S(struct ibv_recv_wr);
    S(struct ibv_wc);
    S(struct ibv_sge);
    S(struct ibv_port_attr);

    printf("== ibv_qp_attr offsets ==\n");
    O(struct ibv_qp_attr, qp_state);
    O(struct ibv_qp_attr, cur_qp_state);
    O(struct ibv_qp_attr, path_mtu);
    O(struct ibv_qp_attr, path_mig_state);
    O(struct ibv_qp_attr, qkey);
    O(struct ibv_qp_attr, rq_psn);
    O(struct ibv_qp_attr, sq_psn);
    O(struct ibv_qp_attr, dest_qp_num);
    O(struct ibv_qp_attr, qp_access_flags);
    O(struct ibv_qp_attr, pkey_index);
    O(struct ibv_qp_attr, alt_pkey_index);
    O(struct ibv_qp_attr, en_sqd_async_notify);
    O(struct ibv_qp_attr, sq_draining);
    O(struct ibv_qp_attr, max_rd_atomic);
    O(struct ibv_qp_attr, max_dest_rd_atomic);
    O(struct ibv_qp_attr, min_rnr_timer);
    O(struct ibv_qp_attr, port_num);
    O(struct ibv_qp_attr, timeout);
    O(struct ibv_qp_attr, retry_cnt);
    O(struct ibv_qp_attr, rnr_retry);
    O(struct ibv_qp_attr, alt_port_num);
    O(struct ibv_qp_attr, alt_timeout);
    O(struct ibv_qp_attr, ah_attr);
    O(struct ibv_qp_attr, alt_ah_attr);

    printf("== ibv_ah_attr / grh offsets ==\n");
    O(struct ibv_ah_attr, dlid);
    O(struct ibv_ah_attr, sl);
    O(struct ibv_ah_attr, src_path_bits);
    O(struct ibv_ah_attr, static_rate);
    O(struct ibv_ah_attr, is_global);
    O(struct ibv_ah_attr, port_num);
    O(struct ibv_ah_attr, grh);
    O(struct ibv_global_route, dgid);
    O(struct ibv_global_route, flow_label);
    O(struct ibv_global_route, sgid_index);
    O(struct ibv_global_route, hop_limit);
    O(struct ibv_global_route, traffic_class);

    printf("== ibv_qp_init_attr offsets ==\n");
    O(struct ibv_qp_init_attr, qp_context);
    O(struct ibv_qp_init_attr, send_cq);
    O(struct ibv_qp_init_attr, recv_cq);
    O(struct ibv_qp_init_attr, srq);
    O(struct ibv_qp_init_attr, cap);
    O(struct ibv_qp_init_attr, qp_type);
    O(struct ibv_qp_init_attr, sq_sig_all);

    printf("== wr / wc offsets ==\n");
    O(struct ibv_send_wr, wr_id);
    O(struct ibv_send_wr, next);
    O(struct ibv_send_wr, sg_list);
    O(struct ibv_send_wr, num_sge);
    O(struct ibv_send_wr, opcode);
    O(struct ibv_send_wr, send_flags);
    O(struct ibv_send_wr, imm_data);
    O(struct ibv_recv_wr, wr_id);
    O(struct ibv_recv_wr, next);
    O(struct ibv_recv_wr, sg_list);
    O(struct ibv_recv_wr, num_sge);
    O(struct ibv_wc, wr_id);
    O(struct ibv_wc, status);
    O(struct ibv_wc, opcode);
    O(struct ibv_wc, byte_len);
    O(struct ibv_wc, imm_data);
    O(struct ibv_wc, qp_num);
    O(struct ibv_wc, wc_flags);
    O(struct ibv_wc, slid);

    printf("== qp_attr_mask values(无条件打印;若编译报 undeclared,\n"
           "   按编译器提示的命名改后重跑) ==\n");
    M(IBV_QP_STATE);
    M(IBV_QP_CUR_STATE);
    M(IBV_QP_EN_SQD_ASYNC_NOTIFY);
    M(IBV_QP_PKEY_INDEX);
    M(IBV_QP_PORT);
    M(IBV_QP_QKEY);
    M(IBV_QP_RQ_PSN);
    M(IBV_QP_SQ_PSN);
    M(IBV_QP_DEST_QPN);       /* 本机命名;标准头文件为 IBV_QP_DEST_QP_NUM */
    M(IBV_QP_ACCESS_FLAGS);
    M(IBV_QP_PATH_MTU);
    M(IBV_QP_TIMEOUT);
    M(IBV_QP_RETRY_CNT);
    M(IBV_QP_RNR_RETRY);
    M(IBV_QP_MAX_QP_RD_ATOMIC);
    M(IBV_QP_MAX_DEST_RD_ATOMIC);
    M(IBV_QP_MIN_RNR_TIMER);
    M(IBV_QP_AV);

    printf("== enums ==\n");
    printf("IBV_QPS_RESET=%d INIT=%d RTR=%d RTS=%d\n",
           IBV_QPS_RESET, IBV_QPS_INIT, IBV_QPS_RTR, IBV_QPS_RTS);
    printf("IBV_QPT_RC=%d\n", IBV_QPT_RC);
    printf("IBV_MTU_4096=%d\n", IBV_MTU_4096);
    printf("IBV_ACCESS_LOCAL_WRITE=0x%x REMOTE_WRITE=0x%x\n",
           IBV_ACCESS_LOCAL_WRITE, IBV_ACCESS_REMOTE_WRITE);
    printf("IBV_WR_SEND=%d\n", IBV_WR_SEND);
    printf("IBV_WC_SUCCESS=%d IBV_WC_RECV=%d IBV_WC_SEND=%d\n",
           IBV_WC_SUCCESS, IBV_WC_RECV, IBV_WC_SEND);
    printf("IBV_SEND_SIGNALED=0x%x\n", IBV_SEND_SIGNALED);
    printf("IBV_LINK_LAYER_ETHERNET=%d\n", IBV_LINK_LAYER_ETHERNET);

    printf("== ibv_port_attr offsets ==\n");
    O(struct ibv_port_attr, state);
    O(struct ibv_port_attr, max_mtu);
    O(struct ibv_port_attr, active_mtu);
    O(struct ibv_port_attr, gid_tbl_len);
    O(struct ibv_port_attr, max_msg_sz);
    O(struct ibv_port_attr, lid);
    O(struct ibv_port_attr, link_layer);
    return 0;
}
