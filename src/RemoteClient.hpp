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
