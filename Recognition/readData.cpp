#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <sstream>

int main() {
    // Initialize a map to store embeddings
    std::map<std::string, std::vector<double>> embeddingsMapStored;

    // Open the embeddings.txt file
    std::ifstream infile("embeddings.txt");
    if (!infile.is_open()) {
        std::cerr << "Failed to open embeddings.txt" << std::endl;
        return 1;
    }

    // Read lines from the file
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string name;

        // Extract name
        if (!(iss >> name)) {
            continue;  // Skip empty lines
        }

        // Get the rest of the line as embeddings
        std::string embeddingString;
        std::getline(iss >> std::ws, embeddingString);
        std::istringstream embeddingStream(embeddingString);

        double val;
        std::vector<double> embeddings;
        while (embeddingStream >> val) {
            embeddings.push_back(val);
        }

        embeddingsMapStored[name] = embeddings;
    }

    // Close the file
    infile.close();
    std::cout<<"Length: "<<embeddingsMapStored.size()<<std::endl;
    // Print the stored embeddings
    for (const auto& pair : embeddingsMapStored) {
        std::cout << pair.first << ": ";
        for (double val : pair.second) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
