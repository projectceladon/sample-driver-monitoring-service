#include <thread>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include "helloworld.grpc.pb.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>
#include <queue>

namespace DMS{
    using namespace objectDetection;
    // Logic and data behind the Server's behavior.
    struct Prediction {
        double tDrowsiness;
        double tDistraction;
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
    };

    class RemoteClient {
        std::unique_ptr<grpc::Server> mServer;
        std::thread mServerThread;
        std::shared_ptr<DetectionServiceImpl> mService;
    public:
        void Shutdown();
        void RunServer();

        std::queue<cv::Mat>& getFrameQueue() {
            return mService->inputFrame;
        }
        void addResult(Prediction r) {
            mService->outputPrediction = r;
            mService->outputPrediction.isValid = true;
        }
        void getLock( ) {
            mService->busy.lock();
        }
        void doUnlock() {
            mService->busy.unlock();
        }
    };
        
}
