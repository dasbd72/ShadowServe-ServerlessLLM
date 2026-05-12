// ----------------------------------------------------------------------------
//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//
//   You may obtain a copy of the License at
//
//                   http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//  ----------------------------------------------------------------------------
#include "tensor_writer.h"

#include <filesystem>
#include <iostream>

TensorWriter::TensorWriter(const std::string& filename) : filename_(filename) {}

TensorWriter::~TensorWriter() {}

uint64_t TensorWriter::writeRecord(const char* data, size_t size) {
  if (partition_idx_ == -1 || partition_size_ + size > kPartitionMaxSize) {
    if (partition_idx_ >= 0) {
      buffer_.reset();
      std::string closed_path =
          filename_ + "_" + std::to_string(partition_idx_);
      closed_file_sizes_.push_back(
          static_cast<uint64_t>(std::filesystem::file_size(closed_path)));
    }
    partition_idx_++;
    partition_size_ = 0;
    std::string partition_filename =
        filename_ + "_" + std::to_string(partition_idx_);
    buffer_ = std::make_unique<AlignedBuffer>(partition_filename);
  }

  // Byte offset in the concatenation of tensor.data_* (what loaders expect),
  // not a pre-rotation logical stream offset (which broke multi-partition
  // checkpoints when a tensor was rotated entirely into the next file).
  uint64_t physical_start = 0;
  for (uint64_t s : closed_file_sizes_) {
    physical_start += s;
  }
  physical_start += static_cast<uint64_t>(partition_size_);

  // make sure the data is 64-bit aligned
  size_t padding = (size % 8) ? (8 - size % 8) : 0;
  size_t written = buffer_->writeData(data, size);
  if (padding) {
    written += buffer_->writePadding(padding);
  }
  partition_size_ += written;

  return physical_start;
}
