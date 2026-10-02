#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/greater_eq.hpp>
#include <openvino/op/less.hpp>
#include <openvino/op/logical_or.hpp>
#include <openvino/op/range.hpp>
#include <openvino/op/select.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/unsqueeze.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

// GGML FILL sets all elements of a tensor to a constant value.
// The constant is stored as a float in op_params[0].
OutputVector translate_fill(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    float c;
    memcpy(&c, context.get_output_op_params(), sizeof(float));

    if (context.get_op_case() == 1 && context.has_input("cache_rs_reset_len")) {
        auto cache_rs_reset_idx = context.get_input("cache_rs_reset_idx");
        auto cache_rs_reset_len = context.get_input("cache_rs_reset_len");
        auto cache_rs = context.get_input(0);

        auto cache_shape = std::make_shared<ov::op::v3::ShapeOf>(cache_rs, ov::element::i64);
        auto n_slots_1d = std::make_shared<ov::op::v8::Gather>(
            cache_shape,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        auto n_slots = std::make_shared<ov::op::v0::Squeeze>(n_slots_1d);

        auto iota = std::make_shared<ov::op::v4::Range>(
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}), n_slots,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}), ov::element::i64);

        auto idx_plus_len = std::make_shared<ov::op::v1::Add>(cache_rs_reset_idx, cache_rs_reset_len);
        auto less_than_idx = std::make_shared<ov::op::v1::Less>(iota, cache_rs_reset_idx);
        auto greater_equal_idx_plus_len = std::make_shared<ov::op::v1::GreaterEqual>(iota, idx_plus_len);
        auto keep_mask = std::make_shared<ov::op::v1::LogicalOr>(less_than_idx, greater_equal_idx_plus_len);
        auto keep_mask_reshape = std::make_shared<ov::op::v0::Unsqueeze>(
            keep_mask, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));

        auto fill_value = ov::op::v0::Constant::create(cache_rs.get_element_type(), ov::Shape{}, {c});
        auto cleared_cache_rs = std::make_shared<ov::op::v1::Select>(keep_mask_reshape, cache_rs, fill_value);
        return rename_outputs_with_suffix({cleared_cache_rs}, context.get_name());
    }

    auto shape = context.get_input_shape(0).to_shape();

    auto val = ov::op::v0::Constant::create(ov::element::f32, {}, {c});
    auto target_shape = ov::op::v0::Constant::create(ov::element::i64, {shape.size()},
        std::vector<int64_t>(shape.begin(), shape.end()));
    auto res = std::make_shared<ov::op::v3::Broadcast>(val, target_shape);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
