import XCTest
@testable import chinese_clip

class BertTokenizerTests: XCTestCase {
    
    var tokenizer: BertTokenizer!

    override func setUpWithError() throws {
        // 在每个测试之前调用
        let bundle = Bundle(for: type(of: self))
        guard let vocabFileURL = bundle.url(forResource: "CoreMLModels/vocab", withExtension: "txt") else {
            XCTFail("Failed to find vocab.txt in the test bundle")
            return
        }
        
        do {
            tokenizer = try BertTokenizer(vocabFileURL: vocabFileURL)
            XCTAssertNotNil(tokenizer, "BertTokenizer should be initialized successfully.")
        } catch {
            XCTFail("Failed to initialize BertTokenizer: \(error)")
        }
    }

    override func tearDownWithError() throws {
        // 在每个测试之后调用
        tokenizer = nil
    }

    func testTokenize() throws {
        let text = "Hello, World!"
        let expectedTokens = ["hello", ",", "world", "!"]
        
        let tokens = tokenizer.tokenize(text: text)
        XCTAssertNotNil(tokens, "Tokens should not be nil")
        XCTAssertEqual(tokens, expectedTokens, "Tokenize function did not return expected tokens.")
    }

    func testTokenizeComplexText() throws {
        let text = "This is a more complex example sentence for tokenization."
        let expectedTokens = ["this", "is", "a", "more", "complex", "example", "sentence", "for", "tokenization", "."]
        
        let tokens = tokenizer.tokenize(text: text)
        XCTAssertNotNil(tokens, "Tokens should not be nil")
        XCTAssertEqual(tokens, expectedTokens, "Tokenize function did not return expected tokens for complex text.")
    }

    func testConvertTokensToIds() throws {
        let tokens = ["hello", ",", "world", "!"]
        
        do {
            let tokenIds = try tokenizer.convertTokensToIds(tokens: tokens)
            XCTAssertNotNil(tokenIds, "Token IDs should not be nil")
            XCTAssertEqual(tokenIds.count, tokens.count, "Number of token IDs does not match number of tokens.")
        } catch {
            XCTFail("convertTokensToIds function threw an error: \(error)")
        }
    }

    func testConvertTokensToIdsWithUnknownToken() throws {
        let tokens = ["hello", "unknown_token", "world", "!"]
        
        do {
            let tokenIds = try tokenizer.convertTokensToIds(tokens: tokens)
            XCTAssertNotNil(tokenIds, "Token IDs should not be nil")
            XCTAssertEqual(tokenIds.count, tokens.count, "Number of token IDs does not match number of tokens, including unknown token.")
        } catch {
            XCTFail("convertTokensToIds function threw an error with unknown token: \(error)")
        }
    }

    func testTokenizeToIds() throws {
        let text = "Hello, World!"
        
        let tokenIds = tokenizer.tokenizeToIds(text: text)
        XCTAssertNotNil(tokenIds, "Token IDs should not be nil")
        XCTAssertGreaterThan(tokenIds.count, 0, "TokenizeToIds function returned no token IDs.")
    }

    func testTokenizeToIdsWithComplexText() throws {
        let text = "This is a more complex example sentence for tokenization."
        
        let tokenIds = tokenizer.tokenizeToIds(text: text)
        XCTAssertNotNil(tokenIds, "Token IDs should not be nil")
        XCTAssertGreaterThan(tokenIds.count, 0, "TokenizeToIds function returned no token IDs for complex text.")
    }

    func testUnTokenize() throws {
        let tokens = ["hello", ",", "world", "!"]
        let tokenIds = try tokenizer.convertTokensToIds(tokens: tokens)
        
        let unTokenized = tokenizer.unTokenize(tokens: tokenIds)
        XCTAssertNotNil(unTokenized, "UnTokenized tokens should not be nil")
        XCTAssertEqual(unTokenized, tokens, "UnTokenize function did not return original tokens.")
    }

    func testRoundTripTokenization() throws {
        let text = "This is a round trip test."
        
        let tokens = tokenizer.tokenize(text: text)
        let tokenIds = try tokenizer.convertTokensToIds(tokens: tokens)
        let unTokenizedTokens = tokenizer.unTokenize(tokens: tokenIds)
        let reconstructedText = tokenizer.convertWordpieceToBasicTokenList(unTokenizedTokens)
        
        XCTAssertNotNil(reconstructedText, "Reconstructed text should not be nil")
        XCTAssertEqual(reconstructedText, text.lowercased(), "Round trip tokenization did not return original text.")
    }
}
