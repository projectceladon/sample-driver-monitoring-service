#include <thread>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include "helloworld.grpc.pb.h"
namespace DMS{
    using namespace objectDetection;
    // Logic and data behind the Server's behavior.
    class DetectionServiceImpl final : public Detection::Service {
        grpc::Status getPredictions(grpc::ServerContext* context, const objectDetection::RequestBytes* request, objectDetection::PredictionsList* response);
    };

    class RemoteClient {
        std::unique_ptr<grpc::Server> mServer;
        std::thread mServerThread;
    public:
        void Shutdown();
        void RunServer();
    };
}