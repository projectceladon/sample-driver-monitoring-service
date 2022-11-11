#include "RemoteClient.hpp"
#include <samples/slog.hpp>

namespace DMS{
    grpc::Status DetectionServiceImpl::getPredictions(grpc::ServerContext* context, const objectDetection::RequestBytes* request, objectDetection::PredictionsList* response) {
        slog::info << "remoteInfer getPredictions" << slog::endl;
        return grpc::Status::OK;
    }

    void RemoteClient::Shutdown() {
        mServer->Shutdown();
        mServerThread.join();
    }

    void RemoteClient::RunServer() {
        mServerThread = std::thread([this]() mutable {
            std::string server_address("0.0.0.0:50051");
            DetectionServiceImpl service;

            grpc::EnableDefaultHealthCheckService(true);
            grpc::reflection::InitProtoReflectionServerBuilderPlugin();
            grpc::ServerBuilder builder;
            // Listen on the given address without any authentication mechanism.
            builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
            // Register "service" as the instance through which we'll communicate with
            // clients. In this case it corresponds to an *synchronous* service.
            builder.RegisterService(&service);
            // Finally assemble the mServer.
            mServer = builder.BuildAndStart();
            std::cout << "Server listening on " << server_address << std::endl;

            // Wait for the mServer to shutdown. Note that some other thread must be
            // responsible for shutting down the mServer for this call to ever return.
            mServer->Wait();
        });
    }
}