//import SwiftUI
//import CoreML
//import CoreImage
//import PhotosUI
//
//struct ContentView: View {
//    @State private var isModelLoaded = false
//    @State private var errorMessage: String?
//    @State private var embedding: MLShapedArray<Float32>?
//    @State private var selectedImage: UIImage?
//    @State private var isImagePickerPresented = false
//
//    var body: some View {
//        VStack {
//            if isModelLoaded {
//                Text("Model Loaded Successfully")
//                    .font(.headline)
//                    .padding()
//
//                Button(action: {
//                    isImagePickerPresented = true
//                }) {
//                    Text("Select Image from Library")
//                        .padding()
//                        .background(Color.blue)
//                        .foregroundColor(.white)
//                        .cornerRadius(10)
//                }
//                .sheet(isPresented: $isImagePickerPresented) {
//                    ImagePicker(image: $selectedImage)
//                }
//                
//                if let selectedImage = selectedImage {
//                    Image(uiImage: selectedImage)
//                        .resizable()
//                        .scaledToFit()
//                        .frame(height: 200)
//                        .padding()
//
//                    Button(action: {
//                        Task {
//                            await computeEmbedding()
//                        }
//                    }) {
//                        Text("Compute Image Embedding")
//                            .padding()
//                            .background(Color.green)
//                            .foregroundColor(.white)
//                            .cornerRadius(10)
//                    }
//                }
//
//                if let embedding = embedding {
//                    Text("Embedding: \(embedding)")
//                        .padding()
//                }
//            } else {
//                Text("Loading Model...")
//                    .font(.headline)
//                    .padding()
//
//                if let errorMessage = errorMessage {
//                    Text("Error: \(errorMessage)")
//                        .foregroundColor(.red)
//                        .padding()
//                }
//            }
//        }
//        .onAppear {
//            Task {
//                await loadModel()
//            }
//        }
//    }
//
//    func loadModel() async {
//        do {
//            let baseURL = URL(fileURLWithPath: Bundle.main.resourcePath!)
//
//            let encoder = try ImgEncoder(resourcesAt: baseURL)
//            isModelLoaded = true
//        } catch {
//            errorMessage = error.localizedDescription
//        }
//    }
//
//    func computeEmbedding() async {
//        guard let sampleImage = selectedImage else {
//            errorMessage = "No image selected"
//            return
//        }
//
//        do {
//            let baseURL = URL(fileURLWithPath: Bundle.main.resourcePath!)
//            let encoder = try ImgEncoder(resourcesAt: baseURL)
//            let embeddingResult = try await encoder.computeImgEmbedding(img: sampleImage)
//            embedding = embeddingResult
//        } catch {
//            errorMessage = error.localizedDescription
//        }
//    }
//}
//
//struct ImagePicker: UIViewControllerRepresentable {
//    @Binding var image: UIImage?
//
//    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
//        let parent: ImagePicker
//
//        init(parent: ImagePicker) {
//            self.parent = parent
//        }
//
//        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
//            if let uiImage = info[.originalImage] as? UIImage {
//                parent.image = uiImage
//            }
//
//            picker.dismiss(animated: true)
//        }
//
//        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
//            picker.dismiss(animated: true)
//        }
//    }
//
//    func makeCoordinator() -> Coordinator {
//        return Coordinator(parent: self)
//    }
//
//    func makeUIViewController(context: Context) -> UIImagePickerController {
//        let picker = UIImagePickerController()
//        picker.delegate = context.coordinator
//        return picker
//    }
//
//    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
//}
//import SwiftUI
//
//struct ContentView: View {
//    var body: some View {
//        WebSocketView()
//    }
//}
//
import SwiftUI

struct ContentView: View {
    var body: some View {
        NavigationView {
            VStack {
                // Assuming WebSocketView has all necessary initializations handled internally
                WebSocketView()

            }
            .navigationTitle("ChineseCLIP")
        }
    }
}
