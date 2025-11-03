#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  // Accumulate across rounds to avoid rebuilding from scratch
  Matrix *K_T_acc = nullptr;   // d x (i+1)
  Matrix *V_stack_acc = nullptr; // (i+1) x d

  for (size_t i = 0; i < keys.size(); ++i) {
    Matrix *Q = rater.GetNextQuery();

    // Move required matrices to Shared Memory (SRAM)
    gpu_sim.MoveMatrixToSharedMem(Q);


    // Update accumulators for K^T and V
    if (i == 0) {
      // Initialize K_T_acc from keys[0]
      gpu_sim.MoveMatrixToSharedMem(keys[0]);
      Matrix *k0_copy = matrix_memory_allocator.Allocate("k_copy_init");
      gpu_sim.Copy(keys[0], k0_copy, kInSharedMemory);
      gpu_sim.Transpose(k0_copy, kInSharedMemory);
      K_T_acc = k0_copy;
      // Initialize V_stack_acc from values[0]
      gpu_sim.MoveMatrixToSharedMem(values[0]);
      Matrix *v0_copy = matrix_memory_allocator.Allocate("v_copy_init");
      gpu_sim.Copy(values[0], v0_copy, kInSharedMemory);
      V_stack_acc = v0_copy;
    } else {
      // Append new key column to K_T_acc
      gpu_sim.MoveMatrixToSharedMem(keys[i]);
      Matrix *k_copy = matrix_memory_allocator.Allocate("k_copy");
      gpu_sim.Copy(keys[i], k_copy, kInSharedMemory);
      gpu_sim.Transpose(k_copy, kInSharedMemory);
      Matrix *K_T_next = matrix_memory_allocator.Allocate("K_T_next");
      gpu_sim.Concat(K_T_acc, k_copy, K_T_next, /*axis=*/1, kInSharedMemory);
      gpu_sim.ReleaseMatrix(K_T_acc);
      gpu_sim.ReleaseMatrix(k_copy);
      K_T_acc = K_T_next;

      // Append new value row to V_stack_acc
      gpu_sim.MoveMatrixToSharedMem(values[i]);
      Matrix *v_copy = matrix_memory_allocator.Allocate("v_copy");
      gpu_sim.Copy(values[i], v_copy, kInSharedMemory);
      Matrix *V_next = matrix_memory_allocator.Allocate("V_next");
      gpu_sim.Concat(V_stack_acc, v_copy, V_next, /*axis=*/0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(V_stack_acc);
      gpu_sim.ReleaseMatrix(v_copy);
      V_stack_acc = V_next;
    }

    // logits = Q * K^T  => shape: (i+1) x (i+1) in SRAM
    // Compute per-row scores on-the-fly to reduce peak memory

    // Build answer row-by-row: for each row, softmax then multiply with V_stack
    Matrix *answer = nullptr;
    for (size_t row_idx = 0; row_idx <= i; ++row_idx) {
      Matrix *q_row = matrix_memory_allocator.Allocate("q_row");
      gpu_sim.GetRow(Q, row_idx, q_row, kInSharedMemory);
      Matrix *scores_row = matrix_memory_allocator.Allocate("scores_row");
      gpu_sim.MatMul(q_row, K_T_acc, scores_row);

      Matrix *row_exp = matrix_memory_allocator.Allocate("row_exp");
      gpu_sim.MatExp(scores_row, row_exp);

      Matrix *row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_exp, row_sum);

      Matrix *row_soft = matrix_memory_allocator.Allocate("row_soft");
      gpu_sim.MatDiv(row_exp, row_sum, row_soft);

      // row_ans = row_soft * V_stack => shape: 1 x d
      Matrix *row_ans = matrix_memory_allocator.Allocate("row_ans");
      gpu_sim.MatMul(row_soft, V_stack_acc, row_ans);

      // Accumulate rows into final answer matrix (vertical concat)
      if (row_idx == 0) {
        answer = matrix_memory_allocator.Allocate("answer_init");
        gpu_sim.Copy(row_ans, answer, kInSharedMemory);
      } else {
        Matrix *answer_next = matrix_memory_allocator.Allocate("answer_next");
        gpu_sim.Concat(answer, row_ans, answer_next, /*axis=*/0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(answer);
        answer = answer_next;
      }

      // Release per-row temporaries to save SRAM
      gpu_sim.ReleaseMatrix(q_row);
      gpu_sim.ReleaseMatrix(scores_row);
      gpu_sim.ReleaseMatrix(row_exp);
      gpu_sim.ReleaseMatrix(row_sum);
      gpu_sim.ReleaseMatrix(row_soft);
      gpu_sim.ReleaseMatrix(row_ans);
    }

    // Move final answer to HBM for committing
    gpu_sim.MoveMatrixToGpuHbm(answer);

    // Release intermediates for this round

    // Execute and commit
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*answer);
    /*
     * Implement your calculation logic here.
     * You can use the GpuSimulator instance to perform matrix operations.
     * For example:
     * gpu_sim.MoveMatrixToGpuHbm(keys[i]);
     * When your need a new matrix, to avoid memory leak, you should use
     * Matrix* new_matrix =
     * matrix_memory_allocator.Allocate(YOUR_MATRIX_NAME(string, which is
     * helpful for debugging)); It can manage the memory of matrices
     * automatically.
     */

    /*
     *
     *
     *
     *
     *
     *
     * YOUR CODE HERE
     *
     *
     *
     *
     *
     *
     */
    gpu_sim.Run(false, &matrix_memory_allocator);
    //rater.CommitAnswer(YOUR_ANSWER_MATRIX)(Commit after running the simulator.)
    /*********************  End of your code *********************/

    /*
     * If you want to print debug information, you can use:
     * gpu_sim.Run(true, &matrix_memory_allocator);
     * At the end of your calculation, you should commit the answer:
     * rater.CommitAnswer(YOUR_ANSWER_MATRIX) in each iteration.
     * Your answer matrix should be in GPU HBM.
     * After the answer is committed, the answer matrix will be released
     * automatically.
     */
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu