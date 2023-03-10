/**
 *
 * Copyright (C) 2018-2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cnn.hpp"

#include <string>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ie_input_info.hpp>
#include <ie_core.hpp>
#include <ie_blob.h>
#include <openvino/openvino.hpp>

using namespace InferenceEngine;

CnnDLSDKBase::CnnDLSDKBase(const Config& config) : config_(config) {}

bool CnnDLSDKBase::Enabled() const {
    return config_.enabled;
}

void CnnDLSDKBase::Load() {
    InferenceEngine::CNNNetwork net_reader = config_.plugin.ReadNetwork(config_.path_to_model);
    const int currentBatchSize = net_reader.getBatchSize();
    if (currentBatchSize != config_.max_batch_size)
        net_reader.setBatchSize(config_.max_batch_size);

    InferenceEngine::InputsDataMap in = net_reader.getInputsInfo();
    if (in.size() != 1) {
        THROW_IE_EXCEPTION << "Network should have only one input";
    }
    in.begin()->second->setPrecision(Precision::U8);
    in.begin()->second->setLayout(Layout::NCHW);
    input_blob_name_ = in.begin()->first;

    OutputsDataMap out = net_reader.getOutputsInfo();
    for (auto&& item : out) {
        item.second->setPrecision(Precision::FP32);
        output_blobs_names_.push_back(item.first);
    }

    executable_network_ = config_.plugin.LoadNetwork(net_reader, config_.deviceName, {});
    infer_request_ = executable_network_.CreateInferRequest();
}

void CnnDLSDKBase::InferBatch(
        const std::vector<cv::Mat>& frames,
        std::function<void(const InferenceEngine::BlobMap&, size_t)> fetch_results) const {
    if (!config_.enabled) {
        return;
    }
    Blob::Ptr input = infer_request_.GetBlob(input_blob_name_);
    const size_t batch_size = input->getTensorDesc().getDims()[0];

    size_t num_imgs = frames.size();
    for (size_t batch_i = 0; batch_i < num_imgs; batch_i += batch_size) {
        const size_t current_batch_size = std::min(batch_size, num_imgs - batch_i);
        for (size_t b = 0; b < current_batch_size; b++) {
            matU8ToBlob<uint8_t>(frames[batch_i + b], input, b);
        }

        if (batch_size != 1)
            infer_request_.SetBatch(current_batch_size);
        infer_request_.Infer();

        InferenceEngine::BlobMap blobs;
        for (const auto& name : output_blobs_names_)  {
            blobs[name] = infer_request_.GetBlob(name);
        }
        fetch_results(blobs, current_batch_size);
    }
}

void CnnDLSDKBase::Infer(const cv::Mat& frame,
                         std::function<void(const InferenceEngine::BlobMap&, size_t)> fetch_results) const {
    InferBatch({frame}, fetch_results);
}

VectorCNN::VectorCNN(const Config& config)
    : CnnDLSDKBase(config) {
    if (config.enabled) {
        Load();
        if (output_blobs_names_.size() != 1) {
            THROW_IE_EXCEPTION << "Demo supports topologies only with 1 output";
        }
    }
}

void VectorCNN::Compute(const cv::Mat& frame,
                                     cv::Mat* vector, cv::Size outp_shape) const {
    std::vector<cv::Mat> output;
    Compute({frame}, &output, outp_shape);
    *vector = output[0];
}

void VectorCNN::Compute(const std::vector<cv::Mat>& images, std::vector<cv::Mat>* vectors,
                                     cv::Size outp_shape) const {
    if (images.empty()) {
        return;
    }
    vectors->clear();
    auto results_fetcher = [vectors, outp_shape](const InferenceEngine::BlobMap& outputs, size_t batch_size) {
        for (auto&& item : outputs) {
            InferenceEngine::Blob::Ptr blob = item.second;
            if (blob == nullptr) {
                THROW_IE_EXCEPTION << "VectorCNN::Compute() Invalid blob '" << item.first << "'";
            }
            InferenceEngine::SizeVector ie_output_dims = blob->getTensorDesc().getDims();
            std::vector<int> blob_sizes(ie_output_dims.size(), 0);
            for (size_t i = 0; i < blob_sizes.size(); ++i) {
                blob_sizes[i] = ie_output_dims[i];
            }
            cv::Mat out_blob(blob_sizes, CV_32F, blob->buffer());
            for (size_t b = 0; b < batch_size; b++) {
                cv::Mat blob_wrapper(out_blob.size[1], 1, CV_32F,
                                     reinterpret_cast<void*>((out_blob.ptr<float>(0) + b * out_blob.size[1])));
                vectors->emplace_back();
                if (outp_shape != cv::Size())
                    blob_wrapper = blob_wrapper.reshape(1, {outp_shape.height, outp_shape.width});
                blob_wrapper.copyTo(vectors->back());
            }
        }
    };
    InferBatch(images, results_fetcher);
}
