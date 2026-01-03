import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Настройка стилей
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataAnalyzer:
    """Класс для анализа данных"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.output_dir = Path("reports/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Загрузка данных"""
        logger.info("Загрузка данных для анализа...")

        # Ищем файлы
        train_path = self.data_dir / "train.csv"
        # test_path = self.data_dir / "test.csv"

        if train_path.exists():
            self.df = pd.read_csv(train_path)
            logger.info(f"Загружен train.csv, размер: {self.df.shape}")
        elif (self.data_dir / "raw").exists():
            raw_files = list((self.data_dir / "raw").glob("*.csv"))
            if raw_files:
                self.df = pd.read_csv(raw_files[0])
                logger.info(f"Загружен сырой файл: {raw_files[0].name}")
            else:
                raise FileNotFoundError("Данные не найдены")
        else:
            raise FileNotFoundError("Данные не найдены")

    def basic_statistics(self):
        """Базовая статистика"""
        logger.info("\n=== Базовая статистика ===")

        # Общая информация
        logger.info(f"Размер данных: {self.df.shape}")
        logger.info(f"\nТипы данных: \n{self.df.dtypes.value_counts()}")

        # Статистика по числовым колонкам
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        logger.info(f"\nЧисловых колонок: {len(numeric_cols)}")

        stats_df = self.df[numeric_cols].describe().T
        stats_df["missing"] = self.df[numeric_cols].isnull().sum()
        stats_df["missing_pct"] = (stats_df["missing"] / len(self.df)) * 100

        logger.info("\nСтатистика по числовым колонкам:")
        logger.info(
            stats_df[["mean", "std", "min", "50%", "max", "missing_pct"]].round(2)
        )

        return stats_df

    def target_distribution(self):
        """Анализ распределения целевой переменной"""
        if "defects" in self.df.columns:
            logger.info("\n=== Распределение целевой переменной ===")

            target_counts = self.df["defects"].value_counts()
            target_pct = self.df["defects"].value_counts(normalize=True) * 100

            logger.info("Количество:")
            logger.info(target_counts)
            logger.info("\nПроценты:")
            logger.info(target_pct.round(2))

            # Визуализация
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Столбчатая диаграмма
            bars = ax1.bar(target_counts.index.astype(str), target_counts.values)
            ax1.set_title("Распределение дефектов (количество)")
            ax1.set_xlabel("Дефект")
            ax1.set_ylabel("Количество")

            # Добавляем значения на столбцы
            for bar in bars:
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height}\n({height / len(self.df) * 100: .1f}%)",
                    ha="center",
                    va="bottom",
                )

            # Круговая диаграмма
            ax2.pie(
                target_counts.values,
                labels=target_counts.index.astype(str),
                autopct="%1.1f%%",
                startangle=90,
            )
            ax2.set_title("Распределение дефектов (проценты)")

            plt.tight_layout()
            plt.savefig(
                self.output_dir / "target_distribution.png",
                dpi=300,
                bbox_inches="tight",  # noqa501
            )
            plt.close()

            return target_counts
        return None

    def correlation_analysis(self):
        """Анализ корреляций"""
        logger.info("\n=== Анализ корреляций ===")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 1:
            # Матрица корреляций
            corr_matrix = self.df[numeric_cols].corr()

            logger.info("\nКорреляция признаков с целевой переменной:")
            if "defects" in corr_matrix.columns:
                target_corr = corr_matrix["defects"].sort_values(ascending=False)
                logger.info(target_corr.round(3))

            # Визуализация матрицы корреляций
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix,
                mask=mask,
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.5},
                annot=False,
                fmt=".2f",
            )
            plt.title("Матрица корреляций")
            plt.tight_layout()
            plt.savefig(
                self.output_dir / "correlation_matrix.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

            # Top коррелирующие признаки
            if "defects" in corr_matrix.columns:
                top_features = (
                    corr_matrix["defects"].abs().sort_values(ascending=False)[1:11]
                )

                plt.figure(figsize=(10, 6))
                top_features.plot(kind="bar")
                plt.title("Top 10 коррелирующих признаков с целевой переменной")
                plt.ylabel("Корреляция (абсолютное значение)")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(
                    self.output_dir / "top_correlations.png",
                    dpi=300,
                    bbox_inches="tight",  # noqa501
                )
                plt.close()

            return corr_matrix
        return None

    def feature_distributions(self, top_n=10):
        """Распределения признаков"""
        logger.info("\n=== Распределения признаков ===")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        # Выбираем top_n признаков с наибольшей дисперсией
        if len(numeric_cols) > top_n:
            variances = self.df[numeric_cols].var().sort_values(ascending=False)
            selected_cols = variances.head(top_n).index.tolist()
        else:
            selected_cols = numeric_cols.tolist()

        logger.info(f"Анализ распределений для {len(selected_cols)} признаков")

        # Создаем subplots
        n_cols = 3
        n_rows = (len(selected_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten()

        for idx, col in enumerate(selected_cols):
            ax = axes[idx]

            # Гистограмма с KDE
            sns.histplot(data=self.df, x=col, ax=ax, kde=True, bins=30)
            ax.set_title(f"Распределение: {col}")
            ax.set_xlabel("")

            # Добавляем статистику
            stats_text = (
                f"mean: {self.df[col].mean(): .2f}\nstd: {self.df[col].std(): .2f}"
            )
            ax.text(
                0.95,
                0.95,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        # Скрываем пустые subplots
        for idx in range(len(selected_cols), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "feature_distributions.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Boxplots для top признаков
        if "defects" in self.df.columns:
            fig, axes = plt.subplots(1, min(5, len(selected_cols)), figsize=(15, 5))
            if min(5, len(selected_cols)) == 1:
                axes = [axes]

            for idx, col in enumerate(selected_cols[:5]):
                sns.boxplot(data=self.df, x="defects", y=col, ax=axes[idx])
                axes[idx].set_title(f"{col} по дефектам")
                axes[idx].set_xlabel("Дефект")

            plt.tight_layout()
            plt.savefig(
                self.output_dir / "boxplots_by_target.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

    def outlier_analysis(self):
        """Анализ выбросов"""
        logger.info("\n=== Анализ выбросов ===")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        # Используем IQR метод для определения выбросов
        outlier_info = {}

        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = self.df[
                (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            ]
            outlier_pct = (len(outliers) / len(self.df)) * 100

            if outlier_pct > 0:
                outlier_info[col] = {
                    "outlier_count": len(outliers),
                    "outlier_pct": outlier_pct,
                    "min": self.df[col].min(),
                    "max": self.df[col].max(),
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                }

        if outlier_info:
            outlier_df = pd.DataFrame(outlier_info).T.sort_values(
                "outlier_pct", ascending=False
            )
            logger.info("\nПризнаки с выбросами: ")
            logger.info(
                outlier_df[["outlier_count", "outlier_pct", "min", "max"]].round(2)
            )

            # Визуализация
            top_outlier_cols = outlier_df.head(10).index.tolist()

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for idx, col in enumerate(top_outlier_cols[:6]):
                ax = axes[idx]
                self.df.boxplot(column=col, ax=ax)
                ax.set_title(f"Выбросы: {col}")
                ax.set_ylabel("Значение")

            for idx in range(len(top_outlier_cols[:6]), len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            plt.savefig(
                self.output_dir / "outliers_boxplots.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

            return outlier_df
        else:
            logger.info("Выбросы не обнаружены")
            return None

    def generate_report(self):
        """Генерация отчета"""
        logger.info("\n=== Генерация отчета EDA ===")

        # Загружаем данные
        self.load_data()

        # Выполняем анализы
        stats_df = self.basic_statistics()
        target_dist = self.target_distribution()
        # corr_matrix = self.correlation_analysis()
        self.feature_distributions()
        outlier_df = self.outlier_analysis()

        # Создаем текстовый отчет
        report_content = []
        report_content.append("=" * 80)
        report_content.append("ОТЧЕТ ПО АНАЛИЗУ ДАННЫХ (EDA)")
        report_content.append("=" * 80)
        report_content.append(f"\nДата генерации: {pd.Timestamp.now()}")
        report_content.append(f"\nРазмер данных: {self.df.shape}")
        report_content.append(f"Количество строк: {self.df.shape[0]}")
        report_content.append(f"Количество признаков: {self.df.shape[1]}")

        if target_dist is not None:
            report_content.append("\nРаспределение целевой переменной: ")
            report_content.append(
                f"Класс 0 (без дефектов): {target_dist.get(0, 0)} "
                f"({target_dist.get(0, 0) / len(self.df) * 100: .1f}%)"
            )
            report_content.append(
                f"Класс 1 (с дефектами): {target_dist.get(1, 0)} "
                f"({target_dist.get(1, 0) / len(self.df) * 100: .1f}%)"
            )

        if stats_df is not None:
            report_content.append("\nПризнаки с пропущенными значениями: ")
            missing_cols = stats_df[stats_df["missing_pct"] > 0]
            if len(missing_cols) > 0:
                for col, row in missing_cols.iterrows():
                    report_content.append(
                        f"  {col}: {row['missing']} пропущенных "
                        f"({row['missing_pct']: .1f}%)"
                    )
            else:
                report_content.append("  Нет пропущенных значений")

        if outlier_df is not None:
            report_content.append("\nПризнаки с выбросами (top 5): ")
            for col, row in outlier_df.head(5).iterrows():
                report_content.append(f"  {col}: {row['outlier_pct']: .1f}% выбросов")

        report_content.append(f"\nДиректория с графиками: {self.output_dir}")
        report_content.append("\n" + "=" * 80)

        # Сохраняем отчет
        report_path = self.output_dir.parent / "eda_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_content))

        logger.info(f"\nОтчет сохранен в {report_path}")
        logger.info(f"Графики сохранены в {self.output_dir}")


def main():
    """Основная функция"""
    # Конфигурация
    DATA_DIR = Path("data/processed")

    analyzer = DataAnalyzer(DATA_DIR)
    analyzer.generate_report()


if __name__ == "__main__":
    main()
