// Public Domain CC0 license. https://creativecommons.org/publicdomain/zero/1.0/
import Tokenizer, { tokenizerFromJson } from './tokenizer.js';

describe('Tokenizer', () => {
  it('should load from JSON', () => {
    const tokenizer = new Tokenizer();
    tokenizer.wordIndex = {
      hello: 1,
      world: 2,
    };
    tokenizer.indexWord = {
      1: 'hello',
      2: 'world'
    };
    const recreated_tokenizer = tokenizerFromJson(tokenizer.toJson());
    expect(recreated_tokenizer.wordIndex).toEqual(tokenizer.wordIndex);
    expect(recreated_tokenizer.indexWord).toEqual(tokenizer.indexWord);
  });

  it('should respect the lower flag', () => {
    const texts = ['hello hello Hello']

    // Test the default assumption
    let tokenizer = new Tokenizer();
    tokenizer.fitOnTexts(texts);
    expect(tokenizer.wordIndex).toEqual({ hello: 1 })

    // Test the lowercase flag
    tokenizer = new Tokenizer({ lower: false });
    tokenizer.fitOnTexts(texts);
    expect(tokenizer.wordIndex).toEqual({ hello: 1, Hello: 2 })
  });

  it('should tokenize texts and store metadata for the texts', () => {
    const tokenizer = new Tokenizer();
    const texts = [
      'hello hello .,/#!$%^&*;:{}= \\ -_`~() hello Hello world world world',
      'great success .,/#!$%^&*;:{}=\\-_`~()   Success'
    ];
    tokenizer.fitOnTexts(texts);
    const sequences = tokenizer.textsToSequences(texts);

    expect(tokenizer.wordIndex).toEqual({
      hello: 1,
      world: 2,
      success: 3,
      great: 4
    });

    expect(tokenizer.indexWord).toEqual({
      1: 'hello',
      2: 'world',
      3: 'success',
      4: 'great'
    });

    expect(tokenizer.wordCounts).toEqual({
      hello: 4,
      world: 3,
      success: 2,
      great: 1
    });

    expect(sequences).toEqual([
      [1, 1, 1, 1, 2, 2, 2],
      [4, 3, 3]
    ]);
  });
});