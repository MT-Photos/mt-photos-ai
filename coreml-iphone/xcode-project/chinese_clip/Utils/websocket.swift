//
//  websocket.swift
//  chinese_clip
//
//  Created by Yixuan Si on 5/23/24.
//
//
import Foundation
import Starscream
import UIKit
import CoreML
//
//// Define custom errors for the WebSocket handling and image processing
enum WebSocketError: Error {
    case connectionError(String)
    case imageDataError
    case encodingError(String)
}
//

class WebSocketImageClient: WebSocketDelegate {
    private var socket: WebSocket?
    private var isConnected = false {
        didSet {
            onConnectionStatusChanged?(isConnected)
        }
    }
    private var imgEncoder: ImgEncoder
    var onConnectionStatusChanged: ((Bool) -> Void)?
    
    var onProcessingCompleted: (() -> Void)?
    
    var clientKey: String?
    private var messageQueue = [String: Data]()
    private var processedMessageIds = Set<String>()
    
    init(serverURL: URL, encoder: ImgEncoder) {
        var request = URLRequest(url: serverURL)
        request.timeoutInterval = 5
        self.imgEncoder = encoder

        socket = WebSocket(request: request)
        socket?.delegate = self
    }

    func connect(withKey key: String) {
        socket?.connect()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            if self?.isConnected == true {
                self?.sendKey(key)
            }
        }
    }

    func disconnect() {
        socket?.disconnect()
        isConnected = false
        print("Disconnected from WebSocket.")
    }

    func didReceive(event: WebSocketEvent, client: WebSocketClient) {
        switch event {
        case .connected(let headers):
            isConnected = true
            print("WebSocket is connected: \(headers)")
        case .disconnected(let reason, let code):
            isConnected = false
            print("WebSocket is disconnected: \(reason) with code: \(code)")
        case .binary(let data):
            handleReceivedData(data)
        case .error(let error):
            print("WebSocket encountered an error: \(String(describing: error))")
        default:
            break
        }
    }

    private func handleReceivedData(_ data: Data) {
        let idSize = 32  // MD5 hash is 32 characters when represented as a string
        let idData = data.prefix(idSize)
        let imageData = data.suffix(from: idSize)
        
        guard let messageId = String(data: idData, encoding: .utf8) else {
            print("Error: Unable to decode message ID")
            return
        }
        
        messageQueue[messageId] = imageData
        processMessagesInOrder()
    }
    
    private func processMessagesInOrder() {
        for (messageId, imageData) in messageQueue {
            if !processedMessageIds.contains(messageId) {
                processImageData(imageData, withId: messageId)
                processedMessageIds.insert(messageId)
                messageQueue.removeValue(forKey: messageId)
            }
        }
    }

    private func processImageData(_ data: Data, withId messageId: String) {
        guard let image = UIImage(data: data) else {
            print("Error: Data cannot be converted to UIImage")
            return
        }

        Task {
            do {
                let embedding = try await MLMultiArray(computeEmbedding(from: image))
                let ocr_result = ocr(in: image)
                sendEmbedding(embedding, ocr_result, withId: messageId)
                onProcessingCompleted?()
            } catch {
                print("Error processing image: \(error)")
            }
        }
    }

//    private func sendEmbedding(_ embedding: MLMultiArray, withId messageId: String) {
//        var embeddingArray = [Float]()
//        for i in 0..<embedding.count {
//            embeddingArray.append(embedding[i].floatValue)
//        }
//        
//        let json: [String: Any] = [
//            "id": messageId,
//            "embedding": embeddingArray
//        ]
//        
//        do {
//            let jsonData = try JSONSerialization.data(withJSONObject: json, options: [])
//            if let jsonString = String(data: jsonData, encoding: .utf8) {
//                socket?.write(string: jsonString) {
//                    print("Embedding data sent successfully as JSON.")
//                    self.processedMessageIds.remove(messageId)
//                }
//            }
//        } catch {
//            print("Error serializing JSON: \(error)")
//        }
//    }
    
    
    private func sendEmbedding(_ embedding: MLMultiArray,_ ocr_result: [String: Any] ,withId messageId: String) {
        var embeddingArray = [Float]()
        for i in 0..<embedding.count {
            embeddingArray.append(embedding[i].floatValue)
        }

        let json: [String: Any] = [
            "id": messageId,
            "embedding": embeddingArray,
            "ocr_result": ocr_result["recognizedTexts"]
        ]

        do {
            let jsonData = try JSONSerialization.data(withJSONObject: json, options: [])
            if let jsonString = String(data: jsonData, encoding: .utf8) {
                socket?.write(string: jsonString) {
                    print("Embedding data sent successfully as JSON.")
                    self.processedMessageIds.remove(messageId)
                }
            }
        } catch {
            print("Error serializing JSON: \(error)")
        }
    }

    
    
    private func sendKey(_ key: String) {
        guard isConnected else {
            print("Connection not established or key not set.")
            return
        }
        socket?.write(string: key) {
            print("Key '\(key)' sent successfully.")
        }
    }

    private func computeEmbedding(from image: UIImage) async throws -> MLShapedArray<Float32> {
        return try await imgEncoder.computeImgEmbedding(img: image)
    }
}
