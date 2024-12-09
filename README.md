# Mini Dollar Trading Strategy v7

An optimized and simplified implementation of a systematic trading strategy for mini dollar futures.

## Overview

This project implements an enhanced trading strategy for mini dollar futures, featuring:

- Market structure analysis
- Advanced signal generation
- Risk management system
- Performance analytics
- Parameter optimization capabilities

## Key Features

- **Market Structure Analysis**: Identifies key market patterns and trends
- **Signal Generation**: Creates trading signals based on multiple technical indicators
- **Risk Management**: Implements position sizing and risk control
- **Performance Analytics**: Calculates key metrics including Sharpe ratio, win rate, and returns
- **Parameter Optimization**: Supports strategy parameter optimization via grid search

## Project Structure

```
mini-dolar-strategy-v7/
├── src/
│   ├── analysis/
│   │   ├── market_structure.py
│   │   ├── signals.py
│   │   ├── volatility.py
│   │   └── risk_management.py
│   ├── data/
│   │   └── loaders/
│   │       └── market_data.py
│   ├── models/
│   │   └── metrics.py
│   └── strategy/
│       └── enhanced_strategy.py
├── config/
│   └── strategy_config.yaml
├── tests/
├── main.py
└── requirements.txt
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Rpinto003/mini-dolar-strategy-v7.git
cd mini-dolar-strategy-v7
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The strategy is configured via `config/strategy_config.yaml`. Key parameters include:

- `lookback_period`: Period for market structure analysis
- `atr_period`: Period for Average True Range calculation
- `risk_free_rate`: Risk-free rate for Sharpe ratio calculation
- `initial_capital`: Starting capital for backtesting
- `max_risk_per_trade`: Maximum risk per trade as a percentage

## Usage

1. Configure your strategy parameters in `config/strategy_config.yaml`
2. Run the strategy:
```bash
python main.py
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/my-new-feature`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
