// #include<bits/stdc++.h>
// using namespace std;

// const int IMAGE_WIDTH = 28;
// const int IMAGE_HEIGHT = 28;
// const int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;

// vector<vector<uint8_t>> load_images(const string& filename) {
//     ifstream file(filename, ios::binary);
//     if (!file.is_open()) {
//         throw runtime_error("Cannot open file: " + filename);
//     }

//     // Read header
//     int32_t magic_number = 0;
//     int32_t number_of_images = 0;
//     int32_t rows = 0;
//     int32_t cols = 0;

//     file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
//     file.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
//     file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
//     file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

//     // Convert from big-endian to little-endian
//     magic_number = __builtin_bswap32(magic_number);
//     number_of_images = __builtin_bswap32(number_of_images);
//     rows = __builtin_bswap32(rows);
//     cols = __builtin_bswap32(cols);

//     if (rows != IMAGE_HEIGHT || cols != IMAGE_WIDTH) {
//         throw runtime_error("Unexpected image size");
//     }

//     // Read the image data
//     vector<vector<uint8_t>> images(number_of_images, vector<uint8_t>(IMAGE_SIZE));
//     for (int i = 0; i < number_of_images; ++i) {
//         file.read(reinterpret_cast<char*>(images[i].data()), IMAGE_SIZE);
//     }

//     file.close();
//     return images;
// }

// vector<uint8_t> load_labels(const string& filename) {
//     ifstream file(filename, ios::binary);
//     if (!file.is_open()) {
//         throw runtime_error("Cannot open file: " + filename);
//     }

//     // Read header
//     int32_t magic_number = 0;
//     int32_t number_of_labels = 0;

//     file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
//     file.read(reinterpret_cast<char*>(&number_of_labels), sizeof(number_of_labels));

//     // Convert from big-endian to little-endian
//     magic_number = __builtin_bswap32(magic_number);
//     number_of_labels = __builtin_bswap32(number_of_labels);

//     // Read the label data
//     vector<uint8_t> labels(number_of_labels);
//     file.read(reinterpret_cast<char*>(labels.data()), number_of_labels);

//     file.close();
//     return labels;
// }

// struct Weight{
//     vector<double> w; // weights

//     Weight(int size){
//         w = vector<double>(size);
//         srand(time(0));
//         for (int i=0; i<size; i++){
//             double random_value = (double)(rand()) / RAND_MAX;
//             double random_number = random_value - 0.5;
//             w[i] = random_number;
//         }
//     }
// };

// struct Bias{
//     double b; // bias

//     Bias(){
//         b = 0.0;
//     }
// };
// struct Layer {
//     vector<Weight> w; // weights
//     vector<Bias> b;   // bias
    
//     Layer(int size, int input_size) {
//         for (int i = 0; i < size; ++i) {
//             w.push_back(Weight(input_size));
//             b.push_back(Bias());
//         }
//     }
// };

// class Model{
//     public:
//         vector<Layer> layers;
//         vector <vector<int>> layerValues;

//         void dense(int size , int input_size){
//             layers.push_back(Layer(size , input_size));
//         }

//         void train(vector<vector<int>> x){ // vector of all flattened images ("features")
//             for(int i =0; i < x.size(); ++i){
//                 forwardPropagation(x[i]);
//                 // backPropagation(x[i]);
//             }
//         }
//     private:
//         void forwardPropagation(vector<int> x){
//             layerValues.push_back(x);
//             for(auto& layer: layers){
//                 int size = layer.w.size();
//                 vector <int> prevLayerValues = layerValues[layerValues.size()-1];
//                 vector <int> currLayerValues(size);
//                 for(int i = 0; i < size; ++i){ // a single neuron has w[i] weights and b[i] bias
//                     int currVal = 0;
//                     for (int j=0; j<layer.w[i].w.size(); j++){
//                         currVal += (layer.w[i].w[j]* prevLayerValues[j]);
//                     }
//                     currVal += layer.b[i].b;
//                     currLayerValues.push_back(currVal);
//                 }
//                 layerValues.push_back(currLayerValues);
//             }
//         }
//         void backPropagation(vector<int> x){

//         }
// };

// int main(){
//     Model model;
//     model.dense(128 , IMAGE_SIZE);
//     model.dense(64 , 128);
//     model.dense(10 , 16);
//     cout << model.layers.size() << '\n';
//     cout << model.layers[0].w.size() << '\n';
//     cout << "Weights of 1st layer 1st node: " << endl;

//     for (int i=0; i<model.layers[0].w[0].w.size(); i++){
//         cout << model.layers[0].w[0].w[i] << " ";
//     }
//     cout << endl;

//     try {
//         // Load images and labels
//         string image_filename = "./archive/train-images-idx3-ubyte/train-images-idx3-ubyte";
//         string label_filename = "./archive/train-labels-idx1-ubyte/train-labels-idx1-ubyte";

//         vector<vector<uint8_t>> images = load_images(image_filename);
//         vector<uint8_t> labels = load_labels(label_filename);

//         cout << "Loaded " << images.size() << " images of size " << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << endl;
//         cout << "Loaded " << labels.size() << " labels" << endl;

//         // Example: display the first image and its label
//         cout << "Label of the first image: " << static_cast<int>(labels[0]) << endl;
//         for (int row = 0; row < IMAGE_HEIGHT; ++row) {
//             for (int col = 0; col < IMAGE_WIDTH; ++col) {
//                 cout << (images[0][row * IMAGE_WIDTH + col] > 128 ? '#' : '.');
//             }
//             cout << endl;
//         }
//     } catch (const exception& e) {
//         cerr << "Error: " << e.what() << endl;
//         return 1;
//     }

//     return 0;
// }




#include<bits/stdc++.h>
using namespace std;

const int IMAGE_WIDTH = 28;
const int IMAGE_HEIGHT = 28;
const int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;

// Function to load images
vector<vector<int>> load_images(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }

    // Read header
    int32_t magic_number = 0;
    int32_t number_of_images = 0;
    int32_t rows = 0;
    int32_t cols = 0;

    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    // Convert from big-endian to little-endian
    magic_number = __builtin_bswap32(magic_number);
    number_of_images = __builtin_bswap32(number_of_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    if (rows != IMAGE_HEIGHT || cols != IMAGE_WIDTH) {
        throw runtime_error("Unexpected image size");
    }

    // Read the image data into a vector of integers (0-255)
    vector<vector<int>> images(number_of_images, vector<int>(IMAGE_SIZE));
    for (int i = 0; i < number_of_images; ++i) {
        vector<uint8_t> buffer(IMAGE_SIZE);  // Read data as uint8_t (unsigned char)
        file.read(reinterpret_cast<char*>(buffer.data()), IMAGE_SIZE);
        for (int j = 0; j < IMAGE_SIZE; ++j) {
            images[i][j] = static_cast<int>(buffer[j]);  // Convert uint8_t to int
        }
    }

    file.close();
    return images;
}

// Function to load labels
vector<int> load_labels(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }

    // Read header
    int32_t magic_number = 0;
    int32_t number_of_labels = 0;

    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&number_of_labels), sizeof(number_of_labels));

    // Convert from big-endian to little-endian
    magic_number = __builtin_bswap32(magic_number);
    number_of_labels = __builtin_bswap32(number_of_labels);

    // Read the label data
    vector<int> labels(number_of_labels);
    vector<uint8_t> buffer(number_of_labels);  // Read as uint8_t
    file.read(reinterpret_cast<char*>(buffer.data()), number_of_labels);

    for (int i = 0; i < number_of_labels; ++i) {
        labels[i] = static_cast<int>(buffer[i]);  // Convert uint8_t to int
    }

    file.close();
    return labels;
}

vector<double> relu(vector<double>& input) {
    vector<double> output(input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = max(0.0, input[i]); // ReLU: max(0, x)
    }

    return output;
}

vector<double> softmax(const vector<double>& input) {
    vector<double> output(input.size());

    // Step 1: Find the maximum input value to prevent overflow in the exponential calculations
    double max_val = *max_element(input.begin(), input.end());

    // Step 2: Compute the exponentials and the sum of exponentials
    double sum = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = exp(input[i] - max_val);  // Subtract max_val for numerical stability
        sum += output[i];
    }

    // Step 3: Normalize by dividing each exponential by the sum of exponentials
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] /= sum;
    }

    return output;
}

vector <double> relu_derivative(vector <double>& input){
    vector <double> output(input.size());
    for (int i=0; i<input.size(); i++){
        output[i] = (input[i]>0) ? 1.0 : 0.0;
    }
    return output;
}

vector<double> leaky_relu_derivative(vector<double>& input) {
    vector<double> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = (input[i] > 0) ? 1.0 : 0.01;
    }
    return output;
}

struct Weight{
    vector<double> w; // weights

    Weight(int size){
        w = vector<double>(size);
        for (int i=0; i<size; i++){
            // double random_value = (double)(rand()) / RAND_MAX;
            // double random_number = random_value - 0.5;
            // w[i] = random_number;
            w[i] = 0;
        }
    }
};

struct Bias{
    double b; // bias

    Bias(){
        // b = (double)(rand()) / RAND_MAX;
        // b -= 0.5;
        b = 0;
    }
};
struct Layer {
    int numNodes;
    vector<Weight> weights; // weights
    vector<Bias> b;   // bias
    
    Layer(int size, int input_size) {
        numNodes = size;
        for (int i = 0; i < size; ++i) {
            weights.push_back(Weight(input_size));
            b.push_back(Bias());
        }
    }
};

class Model{
    public:
        vector<Layer> layers;
        vector <vector<double>> layerValues;

        void dense(int size , int input_size){
            layers.push_back(Layer(size , input_size));
        }

        void train(vector<vector<int>> x, vector <int> y){ // vector of all flattened images ("features")
            cout << "Model training started." << endl;
            // for(int i = 0; i < x.size(); i++){
            for(int i = 0; i < 10000; i++){
                layerValues.clear();
                forwardPropagation(x[i]);
                // cout << "Layer values: " << endl;
                // for (int i=0; i<layerValues.size(); i++){
                //     cout << "Layer " << i << ": " << endl;
                //     for (int j=0; j<layerValues[i].size(); j++){
                //         cout << layerValues[i][j] << " ";
                //     }
                //     cout << endl << endl;
                // }
                backPropagation(x[i], y[i]);
                if ((i+1)%500 == 0){
                    cout << i+1 << " images trained." << endl;
                }
            }
            cout << "Model training completed." << endl;
        }

        int predict(vector<int> x) {
            vector<double> input;
            for (auto px : x) {
                input.push_back(px - 0.0); // Converting int to double (though this is unnecessary, it's kept for clarity)
            }

            for (int l = 0; l < layers.size(); l++) {
                auto layer = layers[l];
                int size = layer.numNodes;
                vector<double> currLayerValues;

                for (int i = 0; i < size; i++) { // A single neuron has w[i] weights and b[i] bias
                    double currVal = 0.0; // Change to double for precision

                    // Perform weighted sum of inputs
                    for (int j = 0; j < layer.weights[i].w.size(); j++) {
                        currVal += (layer.weights[i].w[j] * input[j]);
                    }
                    currVal += layer.b[i].b; // Add bias
                    currLayerValues.push_back(currVal);
                }

                if (l != layers.size() - 1) {
                    // Apply ReLU activation for hidden layers
                    currLayerValues = relu(currLayerValues);
                } else {
                    // Apply softmax activation for the output layer
                    cout << "Raw output values before softmax:" << endl;
                    for (double val : currLayerValues) {
                        cout << val << " ";
                    }
                    cout << endl;

                    currLayerValues = softmax(currLayerValues);
                }

                // The input for the next layer becomes the current layer's output
                input = currLayerValues;
            }

            // Now, input holds the softmax output (probabilities for each class)
            cout << "Softmax probabilities:" << endl;
            for (double prob : input) {
                cout << prob << " ";
            }
            cout << endl;

            // Find the index of the maximum probability (this is the predicted class)
            int predicted_number = max_element(input.begin(), input.end()) - input.begin();
            cout << "Predicted class: " << predicted_number << endl;

            return predicted_number;
        }

    private:

        void forwardPropagation(vector<int> x){
            vector <double> input;
            for (auto px : x){
                input.push_back(px-0.0);
            }
            layerValues.push_back(input);

            for(int l=0; l<layers.size(); l++){
                auto layer = layers[l];
                int size = layer.numNodes;
                vector <double> prevLayerValues = layerValues[l];
                vector <double> currLayerValues;
                for(int i = 0; i < size; i++){ // a single neuron has w[i] weights and b[i] bias
                    int currVal = 0;
                    for (int j=0; j<layer.weights[i].w.size(); j++){
                        currVal += (layer.weights[i].w[j]* prevLayerValues[j]);
                    }
                    currVal += layer.b[i].b;
                    currLayerValues.push_back(currVal);
                }
                // cout << "Current layer Values: " << endl;
                // for (int i=0; i<size; i++){
                //     cout << currLayerValues[i] << " ";
                // }
                // cout << endl;

                vector <double> outputValues;
                if (l!=layers.size()-1){
                    outputValues = relu(currLayerValues);
                }else{
                    outputValues = softmax(currLayerValues);
                }
                layerValues.push_back(outputValues);
            }
        }

        void backPropagation(vector<int> x, int y) {
            vector<double> onehotlabel(10, 0.0);
            onehotlabel[y] = 1.0; // One-hot encoding of the label
            
            double learning_rate = 0.1;

            vector<vector<double>> deltas(layers.size());  // To store delta for each layer

            // Step 1: Compute delta for the output layer
            vector<double> output = layerValues.back();  // Get the final layer output
            deltas.back() = vector<double>(output.size());  // Initialize delta for the output layer
            for (int i = 0; i < output.size(); i++) {
                deltas[layers.size()-1][i] = output[i] - onehotlabel[i];  // Compute delta for output layer
            }
            
            // Step 2: Compute delta for hidden layers (backpropagate)
            for (int l = layers.size() - 2; l >= 0; l--) {  // Iterate through hidden layers backwards
                Layer& layer = layers[l];  // Reference to the current layer
                vector<double>& nextLayerValues = layerValues[l + 1];  // Values of the next layer
                vector<double>& currLayerValues = layerValues[l];  // Values of the current layer
                
                deltas[l] = vector<double>(layer.numNodes);  // Initialize delta for the current layer

                // Get the derivative of the activation function for the next layer
                vector<double> d_activation;
                if (l==layers.size()-2){
                    d_activation = nextLayerValues;
                }else{
                    d_activation = leaky_relu_derivative(nextLayerValues);
                }

                // For each neuron in the current layer
                for (int i = 0; i < layer.numNodes; i++) {
                    double error_sum = 0.0;

                    // Sum the weighted errors from the next layer
                    for (int j = 0; j < layers[l + 1].numNodes; j++) {
                        error_sum += (deltas[l + 1][j] * layers[l + 1].weights[j].w[i]);
                    }

                    // Multiply the error by the activation gradient (ReLU derivative)
                    deltas[l][i] = error_sum * d_activation[i];
                }
            }

            // cout << "Deltas value:" << endl;
            // for (int p=deltas.size()-1; p>=0; p--){
            //     cout << "layer " << (p+1) << ": ";
            //     for (int q=0; q<deltas[p].size(); q++){
            //         cout << deltas[p][q] << " ";
            //     }
            //     cout << endl << endl;
            // }

            // Step 3: Update the weights and biases
            for (int l = 0; l < layers.size(); l++) {
                vector<double>& currLayerValues = layerValues[l];  // Values of the current layer
                
                for (int i = 0; i < layers[l].numNodes; i++) {
                    // Update weights
                    for (int j = 0; j < layers[l].weights[i].w.size(); j++) {
                        // cout << "old: " << layers[l].weights[i].w[j] << endl;
                        // cout << "learning rate: " << learning_rate << ", deltas[l][i]: " << deltas[l][i] << ", currLayerValues[j]: " << currLayerValues[j] << endl;
                        layers[l].weights[i].w[j] -= learning_rate * deltas[l][i] * currLayerValues[j];
                        // cout << "new: " << layers[l].weights[i].w[j] << endl;
                    }

                    // Update biases
                    layers[l].b[i].b -= learning_rate * deltas[l][i];
                }
            }
        }
};

int main(){
    vector<vector<int>> train_images;
    vector<int> train_labels;
    try {
        // Load images and labels
        string image_filename = "./archive/train-images-idx3-ubyte/train-images-idx3-ubyte";
        string label_filename = "./archive/train-labels-idx1-ubyte/train-labels-idx1-ubyte";

        train_images = load_images(image_filename);
        train_labels = load_labels(label_filename);

        cout << "Loaded " << train_images.size() << " images of size " << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << endl;
        cout << "Loaded " << train_labels.size() << " labels" << endl;

        // Example: display the first image and its label
        cout << "Label of the first image: " << static_cast<int>(train_labels[0]) << endl;
        for (int row = 0; row < IMAGE_HEIGHT; ++row) {
            for (int col = 0; col < IMAGE_WIDTH; ++col) {
                cout << (train_images[0][row * IMAGE_WIDTH + col] > 128 ? '#' : '.');
            }
            cout << endl;
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    Model model;
    model.dense(128 , IMAGE_SIZE);
    model.dense(64 , 128);
    model.dense(10 , 64);
    cout << model.layers.size() << '\n';
    cout << model.layers[0].numNodes << '\n';
    cout << "Weights of 1st layer 1st node: " << endl;

    vector <double> firstNodeBefore = model.layers[0].weights[0].w;
    vector <double> secondNodeBefore = model.layers[0].weights[1].w;
    model.train(train_images, train_labels);
    vector <double> firstNodeAfter = model.layers[0].weights[0].w;

    cout << "First layer first node weights before training: " << endl;
    for (int i=0; i<firstNodeBefore.size(); i++){
        cout << firstNodeBefore[i] << " ";
    }
    cout << endl << endl;

    // cout << "First layer second node weights before training: " << endl;
    // for (int i=0; i<secondNodeBefore.size(); i++){
    //     cout << secondNodeBefore[i] << " ";
    // }
    // cout << endl;

    cout << "First layer first node weights after training: " << endl;
    for (int i=0; i<firstNodeAfter.size(); i++){
        cout << firstNodeAfter[i] << " ";
    }
    cout << endl;

    // cout << "Before and after training difference sum: ";
    // double diff = 0.0;
    // for (int i=0; i<firstNodeBefore.size(); i++){
    //     diff += firstNodeBefore[i]-firstNodeAfter[i];
    // }
    // cout << diff << endl;

    int test_num;
    while (true){
        cout << "Enter test image number: ";
        cin >> test_num;
        if (test_num == -1) break;
        cout << "Model prediction: " << model.predict(train_images[test_num]) << endl;
        cout << "Actual number: " << train_labels[test_num] << endl;
    }
    
    return 0;
}