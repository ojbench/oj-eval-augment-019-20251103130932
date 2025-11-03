#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    Matrix *Q = rater.GetNextQuery();

    // Move required matrices to Shared Memory (SRAM)
    gpu_sim.MoveMatrixToSharedMem(Q);
    for (size_t k = 0; k <= i; ++k) {
      gpu_sim.MoveMatrixToSharedMem(keys[k]);
      gpu_sim.MoveMatrixToSharedMem(values[k]);
    }

    // Build K_stack (i+1 x d) in SRAM by vertical concatenation of keys[0..i]
    Matrix *K_stack = matrix_memory_allocator.Allocate("K_stack_0");
    gpu_sim.Copy(keys[0], K_stack, kInSharedMemory);
    for (size_t k = 1; k <= i; ++k) {
      Matrix *K_next = matrix_memory_allocator.Allocate("K_stack_next");
      gpu_sim.Concat(K_stack, keys[k], K_next, /*axis=*/0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(K_stack);
      K_stack = K_next;
    }
    // Transpose to get K^T (d x (i+1)) in SRAM
    gpu_sim.Transpose(K_stack, kInSharedMemory);

    // Build V_stack (i+1 x d) in SRAM by vertical concatenation of values[0..i]
    Matrix *V_stack = matrix_memory_allocator.Allocate("V_stack_0");
    gpu_sim.Copy(values[0], V_stack, kInSharedMemory);
    for (size_t k = 1; k <= i; ++k) {
      Matrix *V_next = matrix_memory_allocator.Allocate("V_stack_next");
      gpu_sim.Concat(V_stack, values[k], V_next, /*axis=*/0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(V_stack);
      V_stack = V_next;
    }

    // logits = Q * K^T  => shape: (i+1) x (i+1) in SRAM
    Matrix *logits = matrix_memory_allocator.Allocate("logits");
    gpu_sim.MatMul(Q, K_stack, logits);

    // Build answer row-by-row: for each row, softmax then multiply with V_stack
    Matrix *answer = nullptr;
    for (size_t row_idx = 0; row_idx <= i; ++row_idx) {
      Matrix *row = matrix_memory_allocator.Allocate("logits_row");
      gpu_sim.GetRow(logits, row_idx, row, kInSharedMemory);

      Matrix *row_exp = matrix_memory_allocator.Allocate("row_exp");
      gpu_sim.MatExp(row, row_exp);

      Matrix *row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_exp, row_sum);

      Matrix *row_soft = matrix_memory_allocator.Allocate("row_soft");
      gpu_sim.MatDiv(row_exp, row_sum, row_soft);

      // row_ans = row_soft * V_stack => shape: 1 x d
      Matrix *row_ans = matrix_memory_allocator.Allocate("row_ans");
      gpu_sim.MatMul(row_soft, V_stack, row_ans);

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
      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(row_exp);
      gpu_sim.ReleaseMatrix(row_sum);
      gpu_sim.ReleaseMatrix(row_soft);
      gpu_sim.ReleaseMatrix(row_ans);
    }

    // Move final answer to HBM for committing
    gpu_sim.MoveMatrixToGpuHbm(answer);

    // Release large intermediates to minimize peak memory
    gpu_sim.ReleaseMatrix(K_stack);
    gpu_sim.ReleaseMatrix(V_stack);
    gpu_sim.ReleaseMatrix(logits);

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