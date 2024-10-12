@pointwise(
    size_hints=[33554432],
    filename=__file__,
    triton_meta={
        "signature": {0: "*fp32", 1: "*fp32", 2: "i32"},
        "device": 0,
        "device_type": "cuda",
        "constants": {},
        "configs": [
            instance_descriptor(
                divisible_by_16=(0, 1, 2),
                equal_to_1=(),
                ids_of_folded_args=(),
                divisible_by_8=(2,),
            )
        ],
    },
    inductor_meta={
        "autotune_hints": set(),
        "kernel_name": "triton_poi_fused_mul_relu_0",
        "mutated_arg_names": ["in_out_ptr0"],
    },
    min_elem_per_thread=0,
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 20000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = tmp1 * tmp1
    tmp3 = tmp2 * tmp1
    ...  # 篇幅原因省略中间的行
    tmp49 = tmp48 * tmp1
    tmp50 = tmp49 * tmp1
    tmp51 = tmp50 * tmp1
    tl.store(in_out_ptr0 + (x0), tmp51, xmask)
