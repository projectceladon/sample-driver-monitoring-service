#include "RemoteClient.hpp"
#include <samples/slog.hpp>

namespace DMS {
    grpc::Status DetectionServiceImpl::sendFrame(grpc::ServerContext* context, const objectDetection::RequestBytes* request, objectDetection::ReplyStatus* response) {
        int bufferLength = request->data().length();
        slog::info << "recieved data of length" << bufferLength <<  slog::endl;
        
        //if rawData length is return Status
        if(!bufferLength) {
            slog::info << "invalid or corrupted frame recieved" <<  slog::endl;
            return grpc::Status::CANCELLED;
        }
        //Process the raw Byte data to convert it into cv::Mat
        std::string data = request->data();
        std::vector<char> vdata(data.begin(), data.end());
        cv::Mat matImg;
        matImg = cv::imdecode(vdata ,cv::IMREAD_UNCHANGED);
        inputFrame.push(matImg);
        frameCount++;
        return grpc::Status::OK;
    }

    grpc::Status DetectionServiceImpl::getPredictions(grpc::ServerContext* context, const objectDetection::RequestString* request, objectDetection::Prediction* response) {
        slog::info << "Prediction result entered" <<  slog::endl;
        busy.lock();
            response->set_tdrowsiness(outputPrediction.tDrowsiness);
            response->set_tdistraction(outputPrediction.tDistraction);
            response->set_blinktotal(outputPrediction.blinkTotal);
            response->set_yawntotal(outputPrediction.yawnTotal);
            response->set_x(outputPrediction.x);
            response->set_y(outputPrediction.y);
            response->set_height(outputPrediction.height);
            response->set_width(outputPrediction.width);
            response->set_isvalid(outputPrediction.isValid);
            response->set_inferencetime(outputPrediction.inferenceTime);
            switch(outputPrediction.distLevel) {
                case 0: response->set_distlevel(Prediction_IS_DIST::Prediction_IS_DIST_NOT_DISTRACTED);
                        break;
                case 1: response->set_distlevel(Prediction_IS_DIST::Prediction_IS_DIST_DISTRACTED);
                        break;
                case 2: response->set_distlevel(Prediction_IS_DIST::Prediction_IS_DIST_DISTRACTED);
                        break;
                default: response->set_distlevel(Prediction_IS_DIST::Prediction_IS_DIST_PHONE);
                        break;
            }
        busy.unlock();
        slog::info << "Prediction result fetched" <<  slog::endl;
        return grpc::Status::OK;
    }

    void RemoteClient::Shutdown() {
        mServer->Shutdown();
        mServerThread.join();
    }

    void RemoteClient::RunServer() {
        std::cout << "RunServer E" << std::endl;
        std::string server_address("0.0.0.0:50051");
        mService = std::make_shared<DetectionServiceImpl>();

        grpc::EnableDefaultHealthCheckService(true);
        grpc::reflection::InitProtoReflectionServerBuilderPlugin();
        grpc::ServerBuilder builder;
        // Listen on the given address without any authentication mechanism.
        builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
        // Register "service" as the instance through which we'll communicate with
        // clients. In this case it corresponds to an *synchronous* service.
        builder.RegisterService(mService.get());
        // Finally assemble the mServer.
        mServer = builder.BuildAndStart();
        std::cout << "Server listening on " << server_address << std::endl;

        // Wait for the mServer to shutdown. Note that some other thread must be
        // responsible for shutting down the mServer for this call to ever return.
        mServerThread = std::thread([this]() mutable {
        mServer->Wait();
        });
    }

    std::queue<cv::Mat>& RemoteClient::getFrameQueue() {
        return mService->inputFrame;
    }

    void RemoteClient::addResult(Prediction r) {
        mService->outputPrediction = r;
    }

    void RemoteClient::getLock() {
        mService->busy.lock();
    }

    void RemoteClient::doUnlock() {
        mService->busy.unlock();
    }

    int RemoteClient::getFrameCount() {
        return mService->frameCount;
    }

    void RemoteClient::getInputFrameFPS(bool& getFPS, bool startRemote ) {
        if(startRemote) {
            auto startTime =  std::chrono::system_clock::now();
            while(getFPS) {
                auto currFrameCount = mService->frameCount;
                cameraInputFPS =  currFrameCount - prevFrameCount;
                prevFrameCount = currFrameCount;
                auto endTime = std::chrono::system_clock::now();
                auto diffTime = endTime -  startTime;
                //show curr remote camera fps every 5sec
                if(diffTime.count() >= 5) {
                    slog::info << "remote camera input FPS: " << cameraInputFPS << slog::endl;
                    startTime =  std::chrono::system_clock::now();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
        }
    }
}
