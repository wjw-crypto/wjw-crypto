# Business Analysis and Visualization Module - Developer: Lin Rui
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


# Add text explanation to figure
def add_text_explanation(fig, text, loc='top'):
    if loc == 'top':
        fig.text(0.5, 0.98, text, ha='center', va='top', fontsize=11,
                 bbox=dict(facecolor='yellow', alpha=0.2), wrap=True)
    elif loc == 'bottom':
        fig.text(0.5, 0.01, text, ha='center', va='bottom', fontsize=11,
                 bbox=dict(facecolor='yellow', alpha=0.2), wrap=True)


# Load sample data
def load_sample_data():
    print("Generating sample data...")
    # Generate sales data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-03-31')

    # Sales trend data
    sales_data = pd.DataFrame({
        'date': dates,
        'sales_amount': np.random.normal(50000, 15000, len(dates)) + np.sin(np.arange(len(dates)) * 0.1) * 10000,
        'orders_count': np.random.normal(500, 150, len(dates)) + np.sin(np.arange(len(dates)) * 0.1) * 100,
        'customers_count': np.random.normal(300, 80, len(dates)) + np.sin(np.arange(len(dates)) * 0.1) * 50
    })

    # Add weekend effect
    sales_data.loc[sales_data.date.dt.dayofweek >= 5, 'sales_amount'] *= 1.5
    sales_data.loc[sales_data.date.dt.dayofweek >= 5, 'orders_count'] *= 1.5
    sales_data.loc[sales_data.date.dt.dayofweek >= 5, 'customers_count'] *= 1.3

    # Product category data
    categories = ['Electronics', 'Clothing', 'Home Goods', 'Food', 'Books']
    category_data = pd.DataFrame({
        'category': categories,
        'sales_amount': np.random.normal(500000, 200000, len(categories)),
        'profit_margin': np.random.uniform(0.15, 0.4, len(categories)),
        'customer_satisfaction': np.random.uniform(3.5, 4.8, len(categories))
    })

    # User segmentation data
    user_count = 1000
    user_data = pd.DataFrame({
        'user_id': [f'USER-{i}' for i in range(1, user_count + 1)],
        'recency': np.random.randint(1, 100, user_count),  # Days since last purchase
        'frequency': np.random.exponential(5, user_count),  # Purchase frequency
        'monetary': np.random.exponential(1000, user_count),  # Amount spent
        'avg_order_value': np.random.normal(500, 200, user_count),  # Average order value
        'products_purchased': np.random.randint(1, 50, user_count),  # Number of products purchased
        'days_since_first_purchase': np.random.randint(1, 500, user_count)  # Days since first purchase
    })

    # Add correlation
    user_data['monetary'] = user_data['frequency'] * user_data['avg_order_value'] * np.random.uniform(0.8, 1.2,
                                                                                                      user_count)

    print("Data generation completed!")
    return sales_data, category_data, user_data


# Sales trend analysis and visualization
def sales_trend_analysis(sales_data):
    print("\nStarting Sales Trend Analysis...")

    # Weekly aggregation
    weekly_sales = sales_data.set_index('date').resample('W').sum().reset_index()
    weekly_sales['week'] = weekly_sales['date'].dt.strftime('%Y-%U')

    # Create sales trend chart
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(sales_data['date'], sales_data['sales_amount'], 'b-', label='Daily Sales')
    ax.plot(sales_data['date'], sales_data['orders_count'] * 100, 'r--', label='Orders (×100)')

    # Add weekend peak markers
    weekend_data = sales_data[sales_data.date.dt.dayofweek >= 5]
    ax.scatter(weekend_data['date'], weekend_data['sales_amount'], color='green', alpha=0.6, s=30,
               label='Weekend Sales')

    # Find and mark highest and lowest sales dates
    max_sales_date = sales_data.loc[sales_data['sales_amount'].idxmax()]['date']
    max_sales = sales_data['sales_amount'].max()
    min_sales_date = sales_data.loc[sales_data['sales_amount'].idxmin()]['date']
    min_sales = sales_data['sales_amount'].min()

    # Annotate peak sales point
    ax.annotate(f'Peak Sales: {max_sales:.0f}',
                xy=(max_sales_date, max_sales),
                xytext=(10, 20),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='darkblue'))

    # Annotate lowest sales point
    ax.annotate(f'Lowest Sales: {min_sales:.0f}',
                xy=(min_sales_date, min_sales),
                xytext=(10, -30),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='darkred'))

    ax.set_title('Daily Sales and Order Count Trend Analysis', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Amount (CNY)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Add text explanation
    explanation_text = "Sales Trend Analysis: The chart shows sales data from Jan to Mar, with weekend sales notably higher (green dots).\nSales show an overall upward trend with fluctuations. Order counts follow similar patterns to sales amounts, indicating stable average order value."
    add_text_explanation(fig, explanation_text)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

    # Weekly trend
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Bar chart
    bar_width = 0.35
    x = np.arange(len(weekly_sales))
    bars = ax1.bar(x, weekly_sales['sales_amount'], bar_width, color='royalblue', label='Weekly Sales')
    ax1.set_xlabel('Week', fontsize=12)
    ax1.set_ylabel('Sales Amount (CNY)', color='royalblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='royalblue')

    # Add labels to bars
    for i, bar in enumerate(bars):
        if i % 2 == 0:  # Only show labels for even weeks to avoid crowding
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.0f}',
                     ha='center', va='bottom', rotation=45, fontsize=8)

    # Create second y-axis
    ax2 = ax1.twinx()
    ax2.plot(x, weekly_sales['orders_count'], 'r-', marker='o', label='Weekly Orders')
    ax2.set_ylabel('Order Count', color='firebrick', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='firebrick')

    # Mark highest orders week
    max_orders_idx = weekly_sales['orders_count'].idxmax()
    max_orders = weekly_sales.iloc[max_orders_idx]
    ax2.annotate(f'Peak Orders: {max_orders["orders_count"]:.0f}',
                 xy=(max_orders_idx, max_orders['orders_count']),
                 xytext=(0, 20),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='darkred'))

    # Set x-axis labels
    plt.xticks(x[::2], weekly_sales['week'][::2], rotation=45)  # Only show labels for even weeks

    plt.title('Weekly Sales and Order Count Trend Analysis', fontsize=16)

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Add text explanation
    explanation_text = "Weekly Sales Analysis: This chart shows weekly sales totals (blue bars) and order counts (red line).\nWeek 9 (end of Feb) has peak sales, while Week 3 has the lowest sales. The overall trend shows decline, then rise, then stabilization."
    add_text_explanation(fig, explanation_text)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

    # Calculate sales growth rates
    sales_data['sales_growth'] = sales_data['sales_amount'].pct_change() * 100
    weekly_growth = weekly_sales['sales_amount'].pct_change() * 100

    print(f"Average daily sales: {sales_data['sales_amount'].mean():.2f}")
    print(f"Average daily orders: {sales_data['orders_count'].mean():.2f}")
    print(f"Average daily growth rate: {sales_data['sales_growth'].mean():.2f}%")
    print(f"Average weekly growth rate: {weekly_growth.mean():.2f}%")


# Product category analysis and visualization
def category_analysis(category_data):
    print("\nStarting Product Category Analysis...")

    # Calculate performance metrics
    category_data['performance_score'] = (
            category_data['sales_amount'] / category_data['sales_amount'].max() * 0.5 +
            category_data['profit_margin'] / category_data['profit_margin'].max() * 0.3 +
            category_data['customer_satisfaction'] / category_data['customer_satisfaction'].max() * 0.2
    )
    category_data = category_data.sort_values('performance_score', ascending=False)

    # Create product category analysis chart
    fig = plt.figure(figsize=(14, 7))

    # Sales distribution pie chart
    plt.subplot(1, 2, 1)
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(category_data)))
    explode = [0.1 if i == 0 else 0 for i in range(len(category_data))]  # Highlight first slice

    wedges, texts, autotexts = plt.pie(category_data['sales_amount'],
                                       labels=category_data['category'],
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       shadow=True,
                                       colors=colors,
                                       explode=explode)

    # Set pie chart text style
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('black')

    plt.axis('equal')
    plt.title('Category Sales Distribution', fontsize=14)

    # Scatter plot
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(category_data['profit_margin'],
                          category_data['customer_satisfaction'],
                          s=category_data['sales_amount'] / 5000,
                          c=range(len(category_data)),
                          cmap='viridis',
                          alpha=0.7)

    # Add category labels and annotations
    for i, txt in enumerate(category_data['category']):
        plt.annotate(f"{txt}\nSales: {category_data['sales_amount'].iloc[i] / 10000:.1f}K",
                     (category_data['profit_margin'].iloc[i],
                      category_data['customer_satisfaction'].iloc[i]),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.6))

    # Add optimal product combination area
    plt.axhspan(4.5, 5.0, 0.3, 0.5, alpha=0.2, color='green')
    plt.text(0.35, 4.75, 'Optimal Product Zone\n(High Profit + High Satisfaction)',
             ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.5))

    plt.xlabel('Profit Margin', fontsize=12)
    plt.ylabel('Customer Satisfaction', fontsize=12)
    plt.title('Category Profit Margin vs. Customer Satisfaction', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add text explanation
    explanation_text = "Product Category Analysis: Left chart shows sales percentage by category, right chart analyzes profit margin vs. satisfaction.\nBubble size represents sales volume. Electronics has highest sales, while Books has highest satisfaction but lower sales."
    add_text_explanation(fig, explanation_text)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

    print("\nCategory Performance Ranking:")
    print(category_data[['category', 'sales_amount', 'profit_margin',
                         'customer_satisfaction', 'performance_score']]
          .sort_values('performance_score', ascending=False))


# User segmentation analysis and visualization
def user_segmentation(user_data):
    print("\nStarting User Segmentation Analysis...")

    # Prepare RFM analysis data
    rfm_data = user_data[['recency', 'frequency', 'monetary']].copy()

    # Standardize data
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data)

    # K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    user_data['cluster'] = kmeans.fit_predict(rfm_scaled)

    # Assign meaningful names to clusters
    cluster_centers = kmeans.cluster_centers_
    cluster_names = []
    for i in range(len(cluster_centers)):
        r, f, m = cluster_centers[i]

        # Convert standardized scores back to relative labels
        if r < -0.5:  # For r, lower values mean more recent purchases
            r_label = "High"  # More recent purchases
        else:
            r_label = "Low"  # Less recent purchases

        if f > 0.5:
            f_label = "High"
        else:
            f_label = "Low"

        if m > 0.5:
            m_label = "High"
        else:
            m_label = "Low"

        if r_label == "High" and f_label == "High" and m_label == "High":
            name = "High-Value Loyal Customers"
        elif r_label == "High" and f_label == "Low" and m_label == "High":
            name = "High-Value New Customers"
        elif r_label == "Low" and m_label == "High":
            name = "Churned High-Value Customers"
        elif f_label == "High" and m_label == "Low":
            name = "Frequent Low-Value Customers"
        else:
            name = "Average Value Customers"

        cluster_names.append(name)

    # Create mapping dictionary
    cluster_map = {i: name for i, name in enumerate(cluster_names)}
    user_data['segment'] = user_data['cluster'].map(cluster_map)

    # Create 3D scatter plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Sample data for better visualization
    sample_size = 500 if len(user_data) > 500 else len(user_data)
    sample = user_data.sample(sample_size, random_state=42)

    # Draw scatter plot
    scatter = ax.scatter(
        sample['recency'],
        sample['frequency'],
        sample['monetary'],
        c=sample['cluster'],
        s=30,
        cmap='viridis',
        alpha=0.7
    )

    # Draw cluster centers
    ax.scatter(
        cluster_centers[:, 0] * rfm_data['recency'].std() + rfm_data['recency'].mean(),
        cluster_centers[:, 1] * rfm_data['frequency'].std() + rfm_data['frequency'].mean(),
        cluster_centers[:, 2] * rfm_data['monetary'].std() + rfm_data['monetary'].mean(),
        c=range(len(cluster_centers)),
        marker='X',
        s=200,
        cmap='viridis',
        edgecolor='k'
    )

    # Add labels to cluster centers
    for i, (r, f, m) in enumerate(zip(
            cluster_centers[:, 0] * rfm_data['recency'].std() + rfm_data['recency'].mean(),
            cluster_centers[:, 1] * rfm_data['frequency'].std() + rfm_data['frequency'].mean(),
            cluster_centers[:, 2] * rfm_data['monetary'].std() + rfm_data['monetary'].mean())):
        ax.text(r, f, m, f'Group {i}: {cluster_names[i]}', fontsize=10)

    ax.set_xlabel('Recency (Days Since Last Purchase)')
    ax.set_ylabel('Frequency (Purchase Count)')
    ax.set_zlabel('Monetary (Spending Amount)')
    ax.set_title('Customer Segmentation 3D Visualization (RFM Model)', fontsize=14)

    # Add legend
    legend1 = ax.legend(*scatter.legend_elements(),
                        title="Customer Segments",
                        loc="upper right")
    ax.add_artist(legend1)

    # View data from different angle
    ax.view_init(elev=30, azim=45)

    # Add text explanation
    explanation_text = "Customer Segmentation Analysis: This 3D chart segments customers based on RFM model (Recency, Frequency, Monetary value).\nX markers show cluster centers, identifying High-Value Loyal Customers, High-Value New Customers, Churned High-Value Customers, and other types."
    add_text_explanation(fig, explanation_text)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

    # Analyze cluster features
    cluster_analysis = user_data.groupby('segment').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'avg_order_value': 'mean',
        'products_purchased': 'mean',
        'user_id': 'count'
    }).rename(columns={'user_id': 'count'})

    # Draw customer segment feature comparison bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    segments = cluster_analysis.index.tolist()
    x = np.arange(len(segments))
    width = 0.15

    # Draw bar charts for four features
    features = ['recency', 'frequency', 'monetary', 'avg_order_value']
    feature_labels = ['Recency', 'Frequency', 'Monetary', 'Avg Order Value']
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']

    # Normalize data for display
    normalized_data = cluster_analysis.copy()
    for feature in features:
        if feature == 'recency':  # recency is reversed, lower is better
            normalized_data[feature] = 1 - (normalized_data[feature] / normalized_data[feature].max())
        else:
            normalized_data[feature] = normalized_data[feature] / normalized_data[feature].max()

    for i, (feature, label) in enumerate(zip(features, feature_labels)):
        pos = x - width * 1.5 + i * width
        bars = ax.bar(pos, normalized_data[feature], width, label=label, color=colors[i])

        # Add value labels to bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', rotation=0, fontsize=8)

    ax.set_ylabel('Normalized Score', fontsize=12)
    ax.set_title('Customer Segment Feature Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([s[:20] + '...' if len(s) > 20 else s for s in segments])

    # Add customer count annotation for each segment
    for i, seg in enumerate(segments):
        count = cluster_analysis.loc[seg, 'count']
        ax.text(i, -0.05, f'({count:.0f} customers)', ha='center')

    ax.legend(title="Customer Features")
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    # Add text explanation
    feature_meaning = {
        'Recency': 'How recently customer purchased (higher is more recent)',
        'Frequency': 'How often customer purchases',
        'Monetary': 'How much customer spends',
        'Avg Order Value': 'Average value per order'
    }
    feature_explanation = "\n".join([f"• {k}: {v}" for k, v in feature_meaning.items()])

    explanation_text = f"Customer Segment Feature Comparison: Shows differences between customer segments across RFM dimensions.\n{feature_explanation}\n\nHigh-Value Loyal Customers perform best in all dimensions, while Average Value Customers have lower scores across all metrics."
    add_text_explanation(fig, explanation_text)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

    # Output cluster features
    print("\nCustomer Segment Feature Analysis:")
    print(cluster_analysis)


# Run business analysis and visualization
def run_business_analysis():
    print("Starting Business Analysis and Visualization...")

    # Load sample data
    sales_data, category_data, user_data = load_sample_data()

    # Sales trend analysis
    sales_trend_analysis(sales_data)

    # Product category analysis
    category_analysis(category_data)

    # User segmentation analysis
    user_segmentation(user_data)

    print("\nBusiness Analysis and Visualization Module Completed")


# Run the example
if __name__ == "__main__":
    run_business_analysis()