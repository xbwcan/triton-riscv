// RUN: triton-shared-opt --triton-to-linalg-experimental="structured-ldst-mode=tensor-first-vector-cpu" %s | FileCheck %s

module {
  tt.func public @tensor_add_2d_tensor_first(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>) {
    %r0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %r1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %a_s = tt.splat %a : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %b_s = tt.splat %b : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %c_s = tt.splat %c : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %r0e = tt.expand_dims %r0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %r1e = tt.expand_dims %r1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %r0b = tt.broadcast %r0e : tensor<4x1xi32> -> tensor<4x4xi32>
    %r1b = tt.broadcast %r1e : tensor<1x4xi32> -> tensor<4x4xi32>
    %c4 = arith.constant 4 : i32
    %c4s = tt.splat %c4 : i32 -> tensor<4x4xi32>
    %r0s = arith.muli %r0b, %c4s : tensor<4x4xi32>
    %idx = arith.addi %r0s, %r1b : tensor<4x4xi32>
    %pa = tt.addptr %a_s, %idx : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %pb = tt.addptr %b_s, %idx : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %pc = tt.addptr %c_s, %idx : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %va = tt.load %pa : tensor<4x4x!tt.ptr<f32>>
    %vb = tt.load %pb : tensor<4x4x!tt.ptr<f32>>
    %sum = arith.addf %va, %vb : tensor<4x4xf32>
    tt.store %pc, %sum : tensor<4x4x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @tensor_add_2d_tensor_first
// CHECK:      %[[A:.*]] = memref.reinterpret_cast
// CHECK:      %[[AT:.*]] = bufferization.to_tensor %[[A]] restrict : memref<4x4xf32, strided<[4, 1]>> to tensor<4x4xf32>
// CHECK:      %[[B:.*]] = memref.reinterpret_cast
// CHECK:      %[[BT:.*]] = bufferization.to_tensor %[[B]] restrict : memref<4x4xf32, strided<[4, 1]>> to tensor<4x4xf32>
// CHECK:      %[[SUM:.*]] = linalg.generic
// CHECK:      %[[C:.*]] = memref.reinterpret_cast
// CHECK:      %[[C_CAST:.*]] = memref.cast %[[C]] : memref<4x4xf32, strided<[4, 1]>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK:      bufferization.materialize_in_destination %[[SUM]] in writable %[[C_CAST]]
// CHECK-NOT:  memref.subview
