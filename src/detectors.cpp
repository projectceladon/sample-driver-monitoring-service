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
#include <samples/slog.hpp>
#include <ie_iextension.h>

#if (OPENVINO_VER==2019)
    #include <ext_list.hpp>
#endif

#include <opencv2/opencv.hpp>

#include "detectors.hpp"

using namespace InferenceEngine;

BaseDetection::BaseDetection(std::string topoName,
                             const std::string &pathToModel,
                             const std::string &deviceForInference,
                             int maxBatch, bool isBatchDynamic, bool isAsync)
    : topoName(topoName), pathToModel(pathToModel), deviceForInference(deviceForInference),
      maxBatch(maxBatch), isBatchDynamic(isBatchDynamic), isAsync(isAsync),
      enablingChecked(false), _enabled(false), plugin(NULL) {
    if (isAsync) {
        slog::info << "Use async mode for " << topoName << slog::endl;
    }
}

BaseDetection::~BaseDetection() {}

ExecutableNetwork* BaseDetection::operator ->() {
    return &net;
}

void BaseDetection::submitRequest() {
    if (!enabled() || request == nullptr) return;
    if (isAsync) {
        request->StartAsync();
    } else {
        request->Infer();
    }
}

void BaseDetection::wait() {
    if (!enabled()|| !request || !isAsync)
        return;
    request->Wait(IInferRequest::WaitMode::RESULT_READY);
}

bool BaseDetection::enabled() const  {
    if (!enablingChecked) {
        _enabled = !pathToModel.empty();
        if (!_enabled) {
            slog::info << topoName << " DISABLED" << slog::endl;
        }
        enablingChecked = true;
    }
    return _enabled;
}

FaceDetection::FaceDetection(const std::string &pathToModel,
                             const std::string &deviceForInference,
                             int maxBatch, bool isBatchDynamic, bool isAsync,
                             double detectionThreshold, bool doRawOutputMessages)
    : BaseDetection("Face Detection", pathToModel, deviceForInference, maxBatch, isBatchDynamic, isAsync),
      detectionThreshold(detectionThreshold), doRawOutputMessages(doRawOutputMessages),
      enquedFrames(0), width(0), height(0), bb_enlarge_coefficient(1.2), resultsFetched(false),
      maxProposalCount(0), objectSize(0){
}

void FaceDetection::submitRequest() {
    if (!enquedFrames) return;
    enquedFrames = 0;
    resultsFetched = false;
    results.clear();
    BaseDetection::submitRequest();
}

void FaceDetection::enqueue(const cv::Mat &frame) {
    if (!enabled()) return;

    if (!request) {
        request = net.CreateInferRequestPtr();
    }

    width = frame.cols;
    height = frame.rows;

    Blob::Ptr  inputBlob = request->GetBlob(input);

    matU8ToBlob<uint8_t>(frame, inputBlob);

    enquedFrames = 1;
}

CNNNetwork FaceDetection::read()  {
    slog::info << "Loading network files for Face Detection" << slog::endl;
    InferenceEngine::Core core = InferenceEngine::Core();
    /** Read network model **/
    InferenceEngine::CNNNetwork netReader = core.ReadNetwork(pathToModel);
    /** Set batch size to 1 **/
    slog::info << "Batch size is set to " << maxBatch << slog::endl;
    netReader.setBatchSize(maxBatch);
    /** Extract model name and load it's weights **/
    std::string binFileName = fileNameNoExt(pathToModel) + ".bin";

    /** Read labels (if any)**/
    std::string labelFileName = fileNameNoExt(pathToModel) + ".labels";

    std::ifstream inputFile(labelFileName);
    std::copy(std::istream_iterator<std::string>(inputFile),
              std::istream_iterator<std::string>(),
              std::back_inserter(labels));
    // -----------------------------------------------------------------------------------------------------

    /** SSD-based network should have one input and one output **/
    // ---------------------------Check inputs ------------------------------------------------------
    slog::info << "Checking Face Detection inputs" << slog::endl;
    InputsDataMap inputInfo(netReader.getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("Face Detection network should have only one input");
    }
    InputInfo::Ptr inputInfoFirst = inputInfo.begin()->second;
    inputInfoFirst->setPrecision(Precision::U8);
    // -----------------------------------------------------------------------------------------------------

    // ---------------------------Check outputs ------------------------------------------------------
    slog::info << "Checking Face Detection outputs" << slog::endl;
    OutputsDataMap outputInfo(netReader.getOutputsInfo());
    if (outputInfo.size() != 1) {
        throw std::logic_error("Face Detection network should have only one output");
    }
    DataPtr& _output = outputInfo.begin()->second;
    output = outputInfo.begin()->first;
    const SizeVector outputDims = _output->getTensorDesc().getDims();
    maxProposalCount = outputDims[2];
    objectSize = outputDims[3];
    if (objectSize != 7) {
        throw std::logic_error("Face Detection network output layer should have 7 as a last dimension");
    }
    if (outputDims.size() != 4) {
        throw std::logic_error("Face Detection network output dimensions not compatible shoulld be 4, but was " +
                               std::to_string(outputDims.size()));
    }
    _output->setPrecision(Precision::FP32);

    slog::info << "Loading Face Detection model to the "<< deviceForInference << " plugin" << slog::endl;
    input = inputInfo.begin()->first;
    return netReader;
}

void FaceDetection::fetchResults() {
    if (!enabled()) return;
    results.clear();
    if (resultsFetched) return;
    resultsFetched = true;
    const float *detections = request->GetBlob(output)->buffer().as<float *>();

    for (int i = 0; i < maxProposalCount; i++) {
        float image_id = detections[i * objectSize + 0];
        Result r;
        r.label = static_cast<int>(detections[i * objectSize + 1]);
        r.confidence = detections[i * objectSize + 2];

        if (r.confidence <= detectionThreshold) {
            continue;
        }

        r.location.x = detections[i * objectSize + 3] * width;
        r.location.y = detections[i * objectSize + 4] * height;
        r.location.width = detections[i * objectSize + 5] * width - r.location.x;
        r.location.height = detections[i * objectSize + 6] * height - r.location.y;

        // Make square and enrlarge face bounding box for more robust operation of face analytics networks
        int bb_width = r.location.width;
        int bb_height = r.location.height;

        int bb_center_x = r.location.x + bb_width / 2;
        int bb_center_y = r.location.y + bb_height / 2;

        int max_of_sizes = std::max(bb_width, bb_height);

        int bb_new_width = bb_enlarge_coefficient * max_of_sizes;
        int bb_new_height = bb_enlarge_coefficient * max_of_sizes;

        r.location.x = bb_center_x - bb_new_width / 2;
        r.location.y = bb_center_y - bb_new_height / 2;

        r.location.width = bb_new_width;
        r.location.height = bb_new_height;

        if (image_id < 0) {
            break;
        }
        if (doRawOutputMessages) {
            std::cout << "[" << i << "," << r.label << "] element, prob = " << r.confidence <<
                         "    (" << r.location.x << "," << r.location.y << ")-(" << r.location.width << ","
                      << r.location.height << ")"
                      << ((r.confidence > detectionThreshold) ? " WILL BE RENDERED!" : "") << std::endl;
        }

        results.push_back(r);
    }
}

HeadPoseDetection::HeadPoseDetection(const std::string &pathToModel,
                                     const std::string &deviceForInference,
                                     int maxBatch, bool isBatchDynamic, bool isAsync)
    : BaseDetection("Head Pose", pathToModel, deviceForInference, maxBatch, isBatchDynamic, isAsync),
      outputAngleR("angle_r_fc"), outputAngleP("angle_p_fc"), outputAngleY("angle_y_fc"), enquedFaces(0) {
}

void HeadPoseDetection::submitRequest()  {
    if (!enquedFaces) return;
    if (isBatchDynamic) {
        request->SetBatch(enquedFaces);
    }
    BaseDetection::submitRequest();
    enquedFaces = 0;
}

void HeadPoseDetection::enqueue(const cv::Mat &face) {
    if (!enabled()) {
        return;
    }
    if (enquedFaces == maxBatch) {
        slog::warn << "Number of detected faces more than maximum(" << maxBatch << ") processed by Head Pose detector" << slog::endl;
        return;
    }
    if (!request) {
        request = net.CreateInferRequestPtr();
    }

    Blob::Ptr inputBlob = request->GetBlob(input);

    matU8ToBlob<uint8_t>(face, inputBlob, enquedFaces);

    enquedFaces++;
}

HeadPoseDetection::Results HeadPoseDetection::operator[] (int idx) const {
    Blob::Ptr  angleR = request->GetBlob(outputAngleR);
    Blob::Ptr  angleP = request->GetBlob(outputAngleP);
    Blob::Ptr  angleY = request->GetBlob(outputAngleY);

    return {angleR->buffer().as<float*>()[idx],
                angleP->buffer().as<float*>()[idx],
                angleY->buffer().as<float*>()[idx]};
}

CNNNetwork HeadPoseDetection::read() {
    slog::info << "Loading network files for Head Pose detection " << slog::endl;
    InferenceEngine::Core core = InferenceEngine::Core();
    // Read network model.
    InferenceEngine::CNNNetwork netReader = core.ReadNetwork(pathToModel);
    // Set maximum batch size.
    netReader.setBatchSize(maxBatch);
    slog::info << "Batch size is set to  " << netReader.getBatchSize() << " for Head Pose Network" << slog::endl;

    // ---------------------------Check inputs ------------------------------------------------------
    slog::info << "Checking Head Pose Network inputs" << slog::endl;
    InputsDataMap inputInfo(netReader.getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("Head Pose topology should have only one input");
    }
    InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
    inputInfoFirst->setPrecision(Precision::U8);
    input = inputInfo.begin()->first;
    // -----------------------------------------------------------------------------------------------------

    // ---------------------------Check outputs ------------------------------------------------------
    slog::info << "Checking Head Pose network outputs" << slog::endl;
    OutputsDataMap outputInfo(netReader.getOutputsInfo());
    if (outputInfo.size() != 3) {
        throw std::logic_error("Head Pose network should have 3 outputs");
    }
    for (auto& output : outputInfo) {
        output.second->setPrecision(Precision::FP32);
    }
    std::map<std::string, bool> layerNames = {
        {outputAngleR, false},
        {outputAngleP, false},
        {outputAngleY, false}
    };

    slog::info << "Loading Head Pose model to the "<< deviceForInference << " plugin" << slog::endl;

    _enabled = true;
    return netReader;
}

void HeadPoseDetection::buildCameraMatrix(int cx, int cy, float focalLength) {
    if (!cameraMatrix.empty()) return;
    cameraMatrix = cv::Mat::zeros(3, 3, CV_32F);
    cameraMatrix.at<float>(0) = focalLength;
    cameraMatrix.at<float>(2) = static_cast<float>(cx);
    cameraMatrix.at<float>(4) = focalLength;
    cameraMatrix.at<float>(5) = static_cast<float>(cy);
    cameraMatrix.at<float>(8) = 1;
}

void HeadPoseDetection::drawAxes(cv::Mat& frame, cv::Point3f cpoint, Results headPose, float scale) {
    double yaw   = headPose.angle_y;
    double pitch = headPose.angle_p;
    double roll  = headPose.angle_r;

    pitch *= CV_PI / 180.0;
    yaw   *= CV_PI / 180.0;
    roll  *= CV_PI / 180.0;

    cv::Matx33f        Rx(1,           0,            0,
                          0,  cos(pitch),  -sin(pitch),
                          0,  sin(pitch),  cos(pitch));
    cv::Matx33f Ry(cos(yaw),           0,    -sin(yaw),
                   0,           1,            0,
                   sin(yaw),           0,    cos(yaw));
    cv::Matx33f Rz(cos(roll), -sin(roll),            0,
                   sin(roll),  cos(roll),            0,
                   0,           0,            1);


    auto r = cv::Mat(Rz*Ry*Rx);
    buildCameraMatrix(frame.cols / 2, frame.rows / 2, 950.0);

    cv::Mat xAxis(3, 1, CV_32F), yAxis(3, 1, CV_32F), zAxis(3, 1, CV_32F), zAxis1(3, 1, CV_32F);

    xAxis.at<float>(0) = 1 * scale;
    xAxis.at<float>(1) = 0;
    xAxis.at<float>(2) = 0;

    yAxis.at<float>(0) = 0;
    yAxis.at<float>(1) = -1 * scale;
    yAxis.at<float>(2) = 0;

    zAxis.at<float>(0) = 0;
    zAxis.at<float>(1) = 0;
    zAxis.at<float>(2) = -1 * scale;

    zAxis1.at<float>(0) = 0;
    zAxis1.at<float>(1) = 0;
    zAxis1.at<float>(2) = 1 * scale;

    cv::Mat o(3, 1, CV_32F, cv::Scalar(0));
    o.at<float>(2) = cameraMatrix.at<float>(0);

    xAxis = r * xAxis + o;
    yAxis = r * yAxis + o;
    zAxis = r * zAxis + o;
    zAxis1 = r * zAxis1 + o;

    cv::Point p1, p2;

    p2.x = static_cast<int>((xAxis.at<float>(0) / xAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
    p2.y = static_cast<int>((xAxis.at<float>(1) / xAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
    cv::line(frame, cv::Point(cpoint.x, cpoint.y), p2, cv::Scalar(0, 0, 255), 2);

    p2.x = static_cast<int>((yAxis.at<float>(0) / yAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
    p2.y = static_cast<int>((yAxis.at<float>(1) / yAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
    cv::line(frame, cv::Point(cpoint.x, cpoint.y), p2, cv::Scalar(0, 255, 0), 2);

    p1.x = static_cast<int>((zAxis1.at<float>(0) / zAxis1.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
    p1.y = static_cast<int>((zAxis1.at<float>(1) / zAxis1.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);

    p2.x = static_cast<int>((zAxis.at<float>(0) / zAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
    p2.y = static_cast<int>((zAxis.at<float>(1) / zAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
    cv::line(frame, p1, p2, cv::Scalar(255, 0, 0), 2);
    cv::circle(frame, p2, 3, cv::Scalar(255, 0, 0), 2);
}

FacialLandmarksDetection::FacialLandmarksDetection(const std::string &pathToModel,
                                                   const std::string &deviceForInference,
                                                   int maxBatch, bool isBatchDynamic, bool isAsync)
    : BaseDetection("Facial Landmarks", pathToModel, deviceForInference, maxBatch, isBatchDynamic, isAsync),
      outputFacialLandmarksBlobName("align_fc3"), enquedFaces(0) {
}

void FacialLandmarksDetection::submitRequest() {
    if (!enquedFaces) return;
    if (isBatchDynamic) {
        request->SetBatch(enquedFaces);
    }
    BaseDetection::submitRequest();
    enquedFaces = 0;
}

void FacialLandmarksDetection::enqueue(const cv::Mat &face) {
    if (!enabled()) {
        return;
    }
    if (enquedFaces == maxBatch) {
        slog::warn << "Number of detected faces more than maximum(" << maxBatch << ") processed by Facial Landmarks detector" << slog::endl;
        return;
    }
    if (!request) {
        request = net.CreateInferRequestPtr();
    }

    Blob::Ptr inputBlob = request->GetBlob(input);

    matU8ToBlob<uint8_t>(face, inputBlob, enquedFaces);

    enquedFaces++;
}

std::vector<float> FacialLandmarksDetection::operator[] (int idx) const {
    std::vector<float> normedLandmarks;

    auto landmarksBlob = request->GetBlob(outputFacialLandmarksBlobName);
    auto n_lm = landmarksBlob->getTensorDesc().getDims()[0];
    const float *normed_coordinates = request->GetBlob(outputFacialLandmarksBlobName)->buffer().as<float *>();

    for (auto i = 0UL; i < n_lm; ++i)
        normedLandmarks.push_back(normed_coordinates[i + n_lm * idx]);

    return normedLandmarks;
}

CNNNetwork FacialLandmarksDetection::read() {
    slog::info << "Loading network files for Facial Landmarks detection " << slog::endl;
    InferenceEngine::Core core = InferenceEngine::Core();
    // Read network model.
    InferenceEngine::CNNNetwork netReader = core.ReadNetwork(pathToModel);
    // Set maximum batch size.
    netReader.setBatchSize(maxBatch);
    slog::info << "Batch size is set to  " << netReader.getBatchSize() << " for Facial Landmarks Network" << slog::endl;

    // ---------------------------Check inputs ------------------------------------------------------
    slog::info << "Checking Facial Landmarks Network inputs" << slog::endl;
    InputsDataMap inputInfo(netReader.getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("Facial Landmarks topology should have only one input");
    }
    InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
    inputInfoFirst->setPrecision(Precision::U8);
    input = inputInfo.begin()->first;
    // -----------------------------------------------------------------------------------------------------

    // ---------------------------Check outputs ------------------------------------------------------
    slog::info << "Checking Facial Landmarks network outputs" << slog::endl;
    OutputsDataMap outputInfo(netReader.getOutputsInfo());
    if (outputInfo.size() != 1) {
        throw std::logic_error("Facial Landmarks network should have only one output");
    }
    for (auto& output : outputInfo) {
        output.second->setPrecision(Precision::FP32);
    }
    std::map<std::string, bool> layerNames = {
        {outputFacialLandmarksBlobName, false}
    };


    slog::info << "Loading Facial Landmarks model to the "<< deviceForInference << " plugin" << slog::endl;

    _enabled = true;
    return netReader;
}


Load::Load(BaseDetection& detector) : detector(detector) {
}

void Load::into(Core & plg, const std::string& deviceName, bool enable_dynamic_batch) const {
    if (detector.enabled()) {
        std::map<std::string, std::string> config;
        if (enable_dynamic_batch) {
            config[PluginConfigParams::KEY_DYN_BATCH_ENABLED] = PluginConfigParams::YES;
        }
        detector.net = plg.LoadNetwork(detector.read(), deviceName, config);
        detector.plugin = &plg;
    }
}

CallStat::CallStat():
    _number_of_calls(0), _total_duration(0.0), _last_call_duration(0.0), _smoothed_duration(-1.0) {
}

double CallStat::getSmoothedDuration() {
    // Additional check is needed for the first frame while duration of the first
    // visualisation is not calculated yet.
    if (_smoothed_duration < 0) {
        auto t = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<ms>(t - _last_call_start).count();
    }
    return _smoothed_duration;
}

double CallStat::getTotalDuration() {
    return _total_duration;
}

void CallStat::calculateDuration() {
    auto t = std::chrono::high_resolution_clock::now();
    _last_call_duration = std::chrono::duration_cast<ms>(t - _last_call_start).count();
    _number_of_calls++;
    _total_duration += _last_call_duration;
    if (_smoothed_duration < 0) {
        _smoothed_duration = _last_call_duration;
    }
    double alpha = 0.1;
    _smoothed_duration = _smoothed_duration * (1.0 - alpha) + _last_call_duration * alpha;
}

void CallStat::setStartTime() {
    _last_call_start = std::chrono::high_resolution_clock::now();
}


void Timer::start(const std::string& name) {
    if (_timers.find(name) == _timers.end()) {
        _timers[name] = CallStat();
    }
    _timers[name].setStartTime();
}

void Timer::finish(const std::string& name) {
    auto& timer = (*this)[name];
    timer.calculateDuration();
}

CallStat& Timer::operator[](const std::string& name) {
    if (_timers.find(name) == _timers.end()) {
        throw std::logic_error("No timer with name " + name + ".");
    }
    return _timers[name];
}
