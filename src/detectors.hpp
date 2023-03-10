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

# pragma once

#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>

#include <inference_engine.hpp>
#include "ocv_common.hpp"
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <ie_iextension.h>

#if (OPENVINO_VER==2019)
    #include <ext_list.hpp>
#endif

#include <opencv2/opencv.hpp>

// -------------------------Generic routines for detection networks-------------------------------------------------

struct BaseDetection {
    InferenceEngine::ExecutableNetwork net;
    InferenceEngine::Core * plugin;
    InferenceEngine::InferRequest::Ptr request;
    std::string topoName;
    std::string pathToModel;
    std::string deviceForInference;
    const int maxBatch;
    bool isBatchDynamic;
    const bool isAsync;
    mutable bool enablingChecked;
    mutable bool _enabled;

    BaseDetection(std::string topoName,
                  const std::string &pathToModel,
                  const std::string &deviceForInference,
                  int maxBatch, bool isBatchDynamic, bool isAsync);

    virtual ~BaseDetection();

    InferenceEngine::ExecutableNetwork* operator ->();
    virtual InferenceEngine::CNNNetwork read() = 0;
    virtual void submitRequest();
    virtual void wait();
    bool enabled() const;
};

struct FaceDetection : BaseDetection {
    struct Result {
        int label;
        float confidence;
        cv::Rect location;
    };

    std::string input;
    std::string output;
    double detectionThreshold;
    bool doRawOutputMessages;
    int maxProposalCount;
    int objectSize;
    int enquedFrames;
    float width;
    float height;
    const float bb_enlarge_coefficient;
    bool resultsFetched;
    std::vector<std::string> labels;
    std::vector<Result> results;

    FaceDetection(const std::string &pathToModel,
                  const std::string &deviceForInference,
                  int maxBatch, bool isBatchDynamic, bool isAsync,
                  double detectionThreshold, bool doRawOutputMessages);

    InferenceEngine::CNNNetwork read() override;
    void submitRequest() override;

    void enqueue(const cv::Mat &frame);
    void fetchResults();
};

struct HeadPoseDetection : BaseDetection {
    struct Results {
        float angle_r;
        float angle_p;
        float angle_y;
    };

    std::string input;
    std::string outputAngleR;
    std::string outputAngleP;
    std::string outputAngleY;
    int enquedFaces;
    cv::Mat cameraMatrix;

    HeadPoseDetection(const std::string &pathToModel,
                      const std::string &deviceForInference,
                      int maxBatch, bool isBatchDynamic, bool isAsync);

    InferenceEngine::CNNNetwork read() override;
    void submitRequest() override;

    void enqueue(const cv::Mat &face);
    Results operator[] (int idx) const;
    void buildCameraMatrix(int cx, int cy, float focalLength);
    void drawAxes(cv::Mat& frame, cv::Point3f cpoint, Results headPose, float scale);
};

struct FacialLandmarksDetection : BaseDetection {
    std::string input;
    std::string outputFacialLandmarksBlobName;
    int enquedFaces;
    std::vector<std::vector<float>> landmarks_results;
    std::vector<cv::Rect> faces_bounding_boxes;

    FacialLandmarksDetection(const std::string &pathToModel,
                             const std::string &deviceForInference,
                             int maxBatch, bool isBatchDynamic, bool isAsync);

    InferenceEngine::CNNNetwork read() override;
    void submitRequest() override;

    void enqueue(const cv::Mat &face);
    std::vector<float> operator[] (int idx) const;
};

struct Load {
    BaseDetection& detector;

    explicit Load(BaseDetection& detector);

    void into(InferenceEngine::Core & plg, const std::string& deviceName, bool enable_dynamic_batch = false) const;
};

class CallStat {
public:
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

    CallStat();

    double getSmoothedDuration();
    double getTotalDuration();
    void calculateDuration();
    void setStartTime();

private:
    size_t _number_of_calls;
    double _total_duration;
    double _last_call_duration;
    double _smoothed_duration;
    std::chrono::time_point<std::chrono::high_resolution_clock> _last_call_start;
};

class Timer {
public:
    void start(const std::string& name);
    void finish(const std::string& name);
    CallStat& operator[](const std::string& name);

private:
    std::map<std::string, CallStat> _timers;
};
