"""
Generate synthetic multilingual phishing and legitimate emails for prototyping.

This module creates a diverse dataset of phishing and legitimate emails in 4 languages:
- English (en)
- Swahili (sw)
- Chinese (zh)
- Russian (ru)

Each sample contains: email_text, language_code, label (0=legitimate, 1=phishing)
"""

import pandas as pd
import random
from pathlib import Path
from typing import List, Tuple, Optional


# ============================================================================
# PHISHING TEMPLATES - Common patterns found in phishing emails
# ============================================================================

PHISHING_TEMPLATES_EN = [
    "Urgent: Verify your {service} account immediately. Click here: {url}",
    "Your {service} password has expired. Update it now: {url}",
    "Suspicious activity detected on your {service} account. Confirm identity: {url}",
    "Dear valued customer, please confirm your {service} login credentials: {url}",
    "WARNING: Your {service} account will be suspended unless you verify: {url}",
    "Action required: Update payment method for your {service} account: {url}",
    "Congratulations! You've won a reward from {service}. Claim it: {url}",
    "Your {service} account has been compromised. Reset password here: {url}",
]

PHISHING_TEMPLATES_SW = [
    "Haraka: Thibitisha akaunti yako ya {service}. Bofya hapa: {url}",
    "Neno lako la {service} limepoteza muda wake. Sasisha sasa: {url}",
    "Shughuli isiyo ya kawaida imegundulika kwenye akaunti yako ya {service}: {url}",
    "Tafadhali thibitisha akaunti yako ya {service}: {url}",
    "ONYO: Akaunti yako ya {service} itasimama isipokuwa uthibitishe: {url}",
    "Haraka: Sasisha njia ya malipo kwa akaunti yako ya {service}: {url}",
    "Hongera! Umeshinda zawadi kutoka {service}. Lipia hapa: {url}",
    "Akaunti yako ya {service} imehacked. Sasisha neno lako: {url}",
]

PHISHING_TEMPLATES_ZH = [
    "紧急：验证您的 {service} 账户。点击此处：{url}",
    "您的 {service} 密码已过期。立即更新：{url}",
    "在您的 {service} 账户上检测到可疑活动。确认身份：{url}",
    "尊敬的客户，请确认您的 {service} 登录凭据：{url}",
    "警告：除非您验证，否则您的 {service} 账户将被暂停：{url}",
    "需要操作：为您的 {service} 账户更新付款方式：{url}",
    "恭喜！您从 {service} 赢得了奖励。立即领取：{url}",
    "您的 {service} 账户已被入侵。在此重置密码：{url}",
]

PHISHING_TEMPLATES_RU = [
    "Срочно: Подтвердите свой аккаунт {service}. Кликните здесь: {url}",
    "Ваш пароль {service} истек. Обновите сейчас: {url}",
    "Обнаружена подозрительная активность на вашем аккаунте {service}: {url}",
    "Пожалуйста, подтвердите ваши учетные данные {service}: {url}",
    "ПРЕДУПРЕЖДЕНИЕ: Ваш аккаунт {service} будет заблокирован: {url}",
    "Срочно: Обновите способ оплаты для {service}: {url}",
    "Поздравляем! Вы выиграли приз от {service}. Получите: {url}",
    "Ваш аккаунт {service} взломан. Переустановите пароль: {url}",
]

# ============================================================================
# LEGITIMATE TEMPLATES - Normal business email patterns
# ============================================================================

LEGITIMATE_TEMPLATES_EN = [
    "Hello, here is your {service} monthly invoice for {month}.",
    "Thank you for using {service}. Here are your account details.",
    "We appreciate your business with {service}. See your receipt below.",
    "Your order from {service} has been shipped and will arrive soon.",
    "Welcome to {service}! Thank you for signing up with us.",
    "Here is your {service} billing information for {month}.",
    "Thank you for contacting {service} support team.",
    "Your {service} subscription has been renewed successfully.",
]

LEGITIMATE_TEMPLATES_SW = [
    "Habari, hii ni invoice yako ya kila mwezi kutoka {service} kwa {month}.",
    "Asante kwa kutumia {service}. Hapa kuna maelezo ya akaunti yako.",
    "Tunakubaliana na biashara yako na {service}. Angalia kuitisha chini.",
    "Amri yako kutoka {service} imetumwa na itafika karibuni.",
    "Karibu {service}! Asante kwa kusajili.",
    "Hii ni taarifa ya kukodi kwa {service} kwa {month}.",
    "Asante kwa kuwasiliana na timu ya {service}.",
    "Usajili wako wa {service} umeboreshwa kwa mafanikio.",
]

LEGITIMATE_TEMPLATES_ZH = [
    "你好，这是您从 {service} 的 {month} 月度发票。",
    "感谢您使用 {service}。以下是您的账户详情。",
    "我们感谢您与 {service} 的业务。请查看下方收据。",
    "您从 {service} 的订单已发货，即将送达。",
    "欢迎来到 {service}！感谢您的注册。",
    "这是您的 {service} {month} 月账单信息。",
    "感谢您联系 {service} 支持团队。",
    "您的 {service} 订阅已成功续订。",
]

LEGITIMATE_TEMPLATES_RU = [
    "Здравствуйте, вот ваш счет от {service} за {month}.",
    "Спасибо за использование {service}. Вот ваши реквизиты учетной записи.",
    "Мы ценим вашу деятельность с {service}. См. квитанцию ниже.",
    "Ваш заказ от {service} отправлен и скоро будет доставлен.",
    "Добро пожаловать в {service}! Спасибо за регистрацию.",
    "Вот ваша информация об абонентской плате {service} за {month}.",
    "Спасибо за обращение в службу поддержки {service}.",
    "Ваша подписка на {service} успешно продлена.",
]

# ============================================================================
# HELPER DATA
# ============================================================================

SERVICES = ["PayPal", "Amazon", "Apple", "Google", "Microsoft", "Bank", "Netflix"]
MONTHS = ["January", "February", "March", "April", "May", "June"]
OBFUSCATED_URLS = [
    "bit.ly/verify2024",
    "tinyurl.com/confirm",
    "short.link/secure",
    "goo.gl/check",
    "am@z0n-verify.ru",
    "payp@l-confirm.tk",
    "g00gle-security.tk",
]

# ============================================================================
# DATA GENERATION FUNCTIONS
# ============================================================================


def generate_phishing_email(language: str) -> str:
    """
    Generate a single phishing email in the specified language.
    
    Args:
        language: Language code ('en', 'sw', 'zh', 'ru')
        
    Returns:
        Generated phishing email text
    """
    # Select template based on language
    templates = {
        "en": PHISHING_TEMPLATES_EN,
        "sw": PHISHING_TEMPLATES_SW,
        "zh": PHISHING_TEMPLATES_ZH,
        "ru": PHISHING_TEMPLATES_RU,
    }
    
    template = random.choice(templates.get(language, PHISHING_TEMPLATES_EN))
    
    # Fill in template variables
    email = template.format(
        service=random.choice(SERVICES),
        url=random.choice(OBFUSCATED_URLS)
    )
    
    return email


def generate_legitimate_email(language: str) -> str:
    """
    Generate a single legitimate email in the specified language.
    
    Args:
        language: Language code ('en', 'sw', 'zh', 'ru')
        
    Returns:
        Generated legitimate email text
    """
    # Select template based on language
    templates = {
        "en": LEGITIMATE_TEMPLATES_EN,
        "sw": LEGITIMATE_TEMPLATES_SW,
        "zh": LEGITIMATE_TEMPLATES_ZH,
        "ru": LEGITIMATE_TEMPLATES_RU,
    }
    
    template = random.choice(templates.get(language, LEGITIMATE_TEMPLATES_EN))
    
    # Fill in template variables
    email = template.format(
        service=random.choice(SERVICES),
        month=random.choice(MONTHS)
    )
    
    return email


def generate_dataset(
    samples_per_language: int = 250,
    languages: Optional[List[str]] = None,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate a complete multilingual phishing/legitimate dataset.
    
    Args:
        samples_per_language: Number of samples to generate per language
        languages: List of language codes to include
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: [email_text, language, label]
        - label: 0 = legitimate, 1 = phishing
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Default to 4 languages
    if languages is None:
        languages = ["en", "sw", "zh", "ru"]
    
    # Storage for dataset
    data = []
    
    # Generate samples for each language
    for language in languages:
        print(f"Generating {language} samples...")
        
        # Generate phishing samples (label=1)
        for _ in range(samples_per_language // 2):
            email = generate_phishing_email(language)
            data.append({
                "email_text": email,
                "language": language,
                "label": 1  # 1 = phishing
            })
        
        # Generate legitimate samples (label=0)
        for _ in range(samples_per_language // 2):
            email = generate_legitimate_email(language)
            data.append({
                "email_text": email,
                "language": language,
                "label": 0  # 0 = legitimate
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle to mix languages and labels
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    print(f"\n✓ Generated {len(df)} total samples")
    print(f"  Distribution:\n{df['label'].value_counts().to_string()}")
    print(f"  Languages:\n{df['language'].value_counts().to_string()}")
    
    return df


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """
    Save dataset to CSV file.
    
    Args:
        df: DataFrame to save
        output_path: Path to save CSV file
    """
    # Create parent directories if they don't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✓ Saved dataset to: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ===========================================================================
def main():
    #Main function to generate and save synthetic dataset#
    
    print("=" * 70)
    print("PHISHER2025: Synthetic Dataset Generator")
    print("=" * 70)
    
    # Generate dataset: 250 samples per language × 4 languages = 1000 total
    dataset = generate_dataset(
        samples_per_language=250,
        languages=["en", "sw", "zh", "ru"],
        random_seed=42
    )
    
    # Save to data/processed/ by default
    output_path = "data/processed/phishing_data.csv"
    save_dataset(dataset, output_path)
    
    print("\n✓ Dataset generation complete!")
    print(f"  Location: {output_path}")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Phishing samples: {(dataset['label'] == 1).sum()}")
    print(f"  Legitimate samples: {(dataset['label'] == 0).sum()}")


if __name__ == "__main__":
    main()


def create_synthetic_dataset(output_path: str, samples_per_language: int = 250, languages: Optional[List[str]] = None, random_seed: int = 42) -> str:
    """
    Convenience wrapper to generate and save a synthetic dataset to a user-specified path.

    This performs light validation on the output path (creates parent directories)
    and will raise exceptions on failure so callers (UI/CLI) can handle errors.

    Args:
        output_path: Full path where the CSV will be written (e.g., 'F:\\phisher_data.csv')
        samples_per_language: Number of samples per language
        languages: Optional list of languages (defaults to ['en','sw','zh','ru'])
        random_seed: Seed for reproducibility

    Returns:
        The output_path written
    """
    # Ensure parent directory exists and is writable
    out_path = Path(output_path)
    if not out_path.parent.exists():
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise OSError(f"Could not create parent directory for {output_path}: {e}")

    # Quick write test (open/close) to fail early if disk/permissions problem
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("")
    except Exception as e:
        raise OSError(f"Cannot write to {output_path}: {e}")

    # Generate and save
    df = generate_dataset(samples_per_language=samples_per_language, languages=languages, random_seed=random_seed)
    save_dataset(df, str(out_path))

    return str(out_path)
