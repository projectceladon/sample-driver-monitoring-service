#include "RemoteClient.hpp"
#include <samples/slog.hpp>

namespace DMS{
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
	// if(inputFrame.size() > 3) {
	// 	inputFrame.pop();
	// }
        inputFrame.push(matImg);
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
}
