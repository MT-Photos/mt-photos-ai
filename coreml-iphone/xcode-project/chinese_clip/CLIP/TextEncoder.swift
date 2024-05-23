////// For licensing see accompanying LICENSE.md file.
////// Copyright (C) 2022 Apple Inc. All Rights Reserved.
////
//import Foundation
//import CoreML
//
//#if os(iOS)
//import UIKit
//#endif
//
///////  A model for encoding text
//public struct TextEncoder {
//
//    /// Text tokenizer
//    var tokenizer: BertTokenizer
//
//    /// Embedding model
//    var model: MLModel
//    
//    init(resourcesAt baseURL: URL,
//         configuration config: MLModelConfiguration = .init()
//    ) throws {
//        let textEncoderURL = baseURL.appending(path: "clip_cn_vit-b-16_fp32.text.mlmodelc")
//        let vocabURL = baseURL.appending(path: "vocab.txt")
//        let compiledURL = compileModel(at: textEncoderURL)
//
//        // Text tokenizer and encoder
//        let tokenizer = try BertTokenizer(vocabFileURL: vocabURL)
//        let textEncoderModel = try MLModel(contentsOf: textEncoderURL, configuration: config)
//        
//        self.tokenizer = tokenizer
//        self.model = textEncoderModel
//    }
//    
//    public func computeTextEmbedding(prompt: String) throws -> MLShapedArray<Float32> {
//        let promptEmbedding = try self.encode(prompt)
//        return promptEmbedding
//    }
//    
//    /**
//    /// Creates text encoder which embeds a tokenized string
//    ///
//    /// - Parameters:
//    ///   - tokenizer: Tokenizer for input text
//    ///   - model: Model for encoding tokenized text
//    public init(tokenizer: BPETokenizer, model: MLModel) {
//        self.tokenizer = tokenizer
//        self.model = model
//    }
//     */
//
//    /// Encode input text/string
//    ///
//    ///  - Parameters:
//    ///     - text: Input text to be tokenized and then embedded
//    ///  - Returns: Embedding representing the input text
//    private func encode(_ text: String) throws -> MLShapedArray<Float32> {
//
//        // Get models expected input length
//        let inputLength = inputShape.last!
//
//        // Tokenize, padding to the expected length
//        var (tokens, ids) = tokenizer.tokenize_with_id(text: text, minCount: inputLength)
//
//        // Truncate if necessary
//        if ids.count > inputLength {
//            tokens = tokens.dropLast(tokens.count - inputLength)
//            ids = ids.dropLast(ids.count - inputLength)
//        }
//
//        // Use the model to generate the embedding
//        return try encode(ids: ids)
//    }
//
//    /// Prediction queue
//    let queue = DispatchQueue(label: "textencoder.predict")
//
//    func encode(ids: [Int]) throws -> MLShapedArray<Float32> {
//        let inputName = inputDescription.name
//        let inputShape = inputShape
//
//        let floatIds = ids.map { Float32($0) }
//        let inputArray = MLShapedArray<Float32>(scalars: floatIds, shape: inputShape)
//        let inputFeatures = try! MLDictionaryFeatureProvider(
//            dictionary: [inputName: MLMultiArray(inputArray)])
//
//        let result = try queue.sync { try model.prediction(from: inputFeatures) }
//        let embeddingFeature = result.featureValue(for: "embOutput")
//        return MLShapedArray<Float32>(converting: embeddingFeature!.multiArrayValue!)
//    }
//
//    var inputDescription: MLFeatureDescription {
//        model.modelDescription.inputDescriptionsByName.first!.value
//    }
//
//    var inputShape: [Int] {
//        inputDescription.multiArrayConstraint!.shape.map { $0.intValue }
//    }
//}
