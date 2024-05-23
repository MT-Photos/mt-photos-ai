import Foundation

enum TokenizerError: Error {
    case tooLong(String)
}

class BertTokenizer: ObservableObject {
    private let basicTokenizer = BasicTokenizer()
    private let wordpieceTokenizer: WordpieceTokenizer
    private let maxLen = 512
    
    private let vocab: [String: Int]
    private let ids_to_tokens: [Int: String]
    
    init(vocabFileURL: URL) throws {
        let vocabTxt = try String(contentsOf: vocabFileURL)
        let tokens = vocabTxt.split(separator: "\n").map { String($0) }
        var vocab: [String: Int] = [:]
        var ids_to_tokens: [Int: String] = [:]
        for (i, token) in tokens.enumerated() {
            vocab[token] = i
            ids_to_tokens[i] = token
        }
        self.vocab = vocab
        self.ids_to_tokens = ids_to_tokens
        self.wordpieceTokenizer = WordpieceTokenizer(vocab: self.vocab)
    }
    
    func tokenize(text: String) -> [String] {
        var tokens: [String] = []
        for token in basicTokenizer.tokenize(text: text) {
            for subToken in wordpieceTokenizer.tokenize(word: token) {
                tokens.append(subToken)
            }
        }
        return tokens
    }
    
    func tokenize_with_id(text: String, minCount: Int? = nil) -> (tokens: [String], tokenIDs: [Int]) {
        let tokens = tokenize(text: text)
        do {
            var tokenIDs = try convertTokensToIds(tokens: tokens)
            
            // Ensure the token count meets the minimum count requirement if specified
            if let minCount = minCount, tokenIDs.count < minCount {
                let padID = vocab["[PAD]"]!
                while tokenIDs.count < minCount {
                    tokenIDs.append(padID)
                }
            }

            return (tokens, tokenIDs)
        } catch {
            print(error)
            return ([], [])
        }
    }


    func convertTokensToIds(tokens: [String]) throws -> [Int] {
        if tokens.count > maxLen {
            throw TokenizerError.tooLong(
                """
                Token indices sequence length is longer than the specified maximum
                sequence length for this BERT model (\(tokens.count) > \(maxLen)). Running this
                sequence through BERT will result in indexing errors
                """
            )
        }
        return tokens.map { vocab[$0]! }
    }
    
    /// Main entry point
    func tokenizeToIds(text: String) -> [Int] {
        return try! convertTokensToIds(tokens: tokenize(text: text))
    }
    
    func tokenToId(token: String) -> Int {
        return vocab[token]!
    }
    
    /// Un-tokenization: get tokens from tokenIds
    func unTokenize(tokens: [Int]) -> [String] {
        return tokens.map { ids_to_tokens[$0]! }
    }
    
    /// Un-tokenization:
    func convertWordpieceToBasicTokenList(_ wordpieceTokenList: [String]) -> String {
        var tokenList: [String] = []
        var individualToken: String = ""
        
        for token in wordpieceTokenList {
            if token.starts(with: "##") {
                individualToken += String(token.suffix(token.count - 2))
            } else {
                if individualToken.count > 0 {
                    tokenList.append(individualToken)
                }
                
                individualToken = token
            }
        }
        
        tokenList.append(individualToken)
        
        return tokenList.joined(separator: " ")
    }
}

class BasicTokenizer {
    let neverSplit = [
        "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"
    ]
    
    func tokenize(text: String) -> [String] {
        let splitTokens = text.folding(options: .diacriticInsensitive, locale: nil)
            .components(separatedBy: NSCharacterSet.whitespaces)
        let tokens = splitTokens.flatMap({ (token: String) -> [String] in
            if neverSplit.contains(token) {
                return [token]
            }
            var toks: [String] = []
            var currentTok = ""
            for c in token.lowercased() {
                if c.isLetter || c.isNumber || c == "°" {
                    currentTok += String(c)
                } else if currentTok.count > 0 {
                    toks.append(currentTok)
                    toks.append(String(c))
                    currentTok = ""
                } else {
                    toks.append(String(c))
                }
            }
            if currentTok.count > 0 {
                toks.append(currentTok)
            }
            return toks
        })
        return tokens
    }
}

class WordpieceTokenizer {
    private let unkToken = "[UNK]"
    private let maxInputCharsPerWord = 100
    private let vocab: [String: Int]
    
    init(vocab: [String: Int]) {
        self.vocab = vocab
    }
    
    static func substr(_ s: String, _ r: Range<Int>) -> String? {
        let stringCount = s.count
        if stringCount < r.upperBound || stringCount < r.lowerBound {
            return nil
        }
        let startIndex = s.index(s.startIndex, offsetBy: r.lowerBound)
        let endIndex = s.index(startIndex, offsetBy: r.upperBound - r.lowerBound)
        return String(s[startIndex..<endIndex])
    }

    /// `word`: A single token.
    /// Warning: this differs from the `pytorch-transformers` implementation.
    /// This should have already been passed through `BasicTokenizer`.
    func tokenize(word: String) -> [String] {
        if word.count > maxInputCharsPerWord {
            return [unkToken]
        }
        var outputTokens: [String] = []
        var isBad = false
        var start = 0
        var subTokens: [String] = []
        while start < word.count {
            var end = word.count
            var cur_substr: String? = nil
            while start < end {
                var substr = WordpieceTokenizer.substr(word, start..<end)!
                if start > 0 {
                    substr = "##\(substr)"
                }
                if vocab[substr] != nil {
                    cur_substr = substr
                    break
                }
                end -= 1
            }
            if cur_substr == nil {
                isBad = true
                break
            }
            subTokens.append(cur_substr!)
            start = end
        }
        if isBad {
            outputTokens.append(unkToken)
        } else {
            outputTokens.append(contentsOf: subTokens)
        }
        return outputTokens
    }
}
