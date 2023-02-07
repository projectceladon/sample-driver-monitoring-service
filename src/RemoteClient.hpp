/**
 * Copyright (C) 2022-2023 Intel Corporation
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

#include <thread>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>
#include <queue>
#include "dms.grpc.pb.h"
#include "detectors.hpp"



namespace DMS {
    using namespace objectDetection;
    // Logic and data behind the Server's behavior.
    struct Prediction {
        double tDrowsiness;
        double tDistraction;
        double inferenceTime;
        int blinkTotal;
        int yawnTotal;
        int distLevel;
        float x;
        float y;
        float height;
        float width;
        bool isValid = false;
    };
    class DetectionServiceImpl : public Detection::Service {
        grpc::Status sendFrame(grpc::ServerContext* context, const objectDetection::RequestBytes* request, objectDetection::ReplyStatus* response);
        grpc::Status getPredictions(grpc::ServerContext* context, const objectDetection::RequestString* request, objectDetection::Prediction* response);
        public:
            std::queue<cv::Mat> inputFrame;
            Prediction outputPrediction;
            std::mutex busy;
            int frameCount = 0;
    };

    class RemoteClient {
        std::unique_ptr<grpc::Server> mServer;
        std::thread mServerThread;
        std::shared_ptr<DetectionServiceImpl> mService;
        int prevFrameCount = 0;
        int cameraInputFPS = 0;
    public:
        void Shutdown();
        void RunServer();
        std::queue<cv::Mat>& getFrameQueue();
        void addResult(Prediction r);
        void getLock();
        void doUnlock();
        int getFrameCount();
        void getInputFrameFPS(bool& getFPS, bool startRemote );
    };
        
}
