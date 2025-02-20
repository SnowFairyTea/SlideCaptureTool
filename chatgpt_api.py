from openai import OpenAI

def refine_transcription(raw_text: str, model: str) -> str:
    api_key = None
    with open('openai_apikey', 'r') as file:
        api_key = file.read().strip()
    
    if not api_key or len(raw_text) < 100:
        return raw_text
    openai_client = OpenAI(api_key=api_key)

    """
    文字起こしの生データを、LLMを利用して読みやすい文章に変換する関数
    """

    # プロンプト
    prompt = f"次の文字起こしデータを、読みやすく自然な日本語の文章に直してください。:\n\n{raw_text}"

    try:
        response = openai_client.chat.completions.create(model=model,  # モデルを引数として受け取る
        messages=[
            {"role": "system", "content": "あなたはプロの文章校正者です。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1024)
        refined_text = response.choices[0].message.content
    except Exception as e:
        print("LLMによる変換中にエラーが発生:", e)
        refined_text = raw_text  # エラー時は生データを返すなど
    return refined_text
