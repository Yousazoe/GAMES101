#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1,
        -eye_pos[2], 0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.

    float angle = rotation_angle / 180.0 * MY_PI;

    Eigen::Matrix4f rotation = Eigen::Matrix4f::Identity();
    rotation << std::cos(angle), -std::sin(angle), 0, 0,
                std::sin(angle), std::cos(angle), 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1;

    model = model * rotation;

    return model;
}

Eigen::Matrix4f get_rotation(Vector3f axis,float rotation_angle)
{
    Eigen::Matrix4f I, N;
    Eigen::Matrix4f rodrigues = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the any axis.
    // Then return it.

    Eigen::Vector4f n;
    Eigen::RowVector4f nT;
    float angle = rotation_angle / 180.0 * MY_PI;

    n  << axis.x(), axis.y(), axis.z(), 0;
    nT << axis.x(), axis.y(), axis.z(), 0;

    I << 1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1;

    N <<   0,    -n.z(), n.y(), 0,
          n.z(),   0,   -n.x(), 0,
         -n.y(), n.x(),   0,    0,
           0,     0,      0,    1;


    rodrigues = std::cos(angle) * I + (1 - std::cos(angle)) * n * nT + std::sin(angle) * N;
    rodrigues(3, 3) = 1;
    return rodrigues;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    // Students will implement this function

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    // Then return it.

    // Matrix ortho

    float angle = eye_fov / 180.0 * MY_PI;

    float top = zNear * std::tan(angle / 2);
    float bot = -top;
    float left = top * aspect_ratio;
    float right = -left;
    float near = -zNear;
    float far = -zFar;

    Eigen::Matrix4f ortho = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f trans(4,4);
    Eigen::Matrix4f scale(4,4);

    trans << 2 / (right - left), 0, 0, 0,
             0, 2 / (top - bot), 0, 0,
             0, 0, 2 / (near -far), 0,
             0, 0, 0, 1;

    scale << 1, 0, 0, -(right + left) / 2,
             0, 1, 0, -(top + bot) / 2,
             0, 0, 1, -(near + far) / 2,
             0, 0, 0, 1;

    ortho = trans * scale;

    // Matrix persp2ortho

    float A = near + far;
    float B = -near * far;

    Eigen::Matrix4f persp2ortho = Eigen::Matrix4f::Identity();

    persp2ortho << near, 0, 0, 0,
                   0, near, 0, 0,
                   0, 0, A, B,
                   0, 0, 1, 0;


    // Matrix presp
    projection = ortho * persp2ortho;

    return projection;
}

int main(int argc, const char** argv)
{
    float angle = 0;
    bool command_line = false;

    Vector3f axis(0,0,1);
    std::string filename = "output.png";

    float angleR = 0, inputAngle = 0;
    Eigen::Vector3f axisR(0,0,1);

    if (argc >= 3) {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        }
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0, 0, 5};

    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }


    bool flagR = false;
    std::cout << "Please enter the axis and angle: " << std::endl;
    std::cin >> axisR.x() >> axisR.y() >> axisR.z() >> inputAngle;

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        // r.set_model(get_model_matrix(axis,angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        if (flagR)
            r.set_rodrigues(get_rotation(axisR, angleR));
        else
            r.set_rodrigues(get_rotation(axis, 0));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a') {
            angle += 10;
        }
        else if (key == 'd') {
            angle -= 10;
        }
        else if (key == 'r') {
            flagR = true;
            angleR += inputAngle;
        }
    }

    return 0;
}
