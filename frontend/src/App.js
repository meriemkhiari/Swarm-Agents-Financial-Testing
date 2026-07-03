import React, { useState } from 'react';

const FIELD_LABEL_STYLE = {
  display: 'block',
  fontWeight: 500,
  marginBottom: '8px',
  color: '#8a8a86',
  fontSize: '14px',
  textTransform: 'uppercase',
  letterSpacing: '0.5px'
};

const INPUT_STYLE = {
  width: '100%',
  padding: '16px 20px',
  background: 'rgba(255, 255, 255, 0.04)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
  borderRadius: '12px',
  fontSize: '15px',
  color: '#f0f0ee',
  outline: 'none',
  transition: 'border-color 0.2s ease'
};

const GRID_STYLE = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
  gap: '20px',
  marginBottom: '24px'
};

const FIELDS = [
  { name: 'user_id', label: 'User ID', type: 'text' },
  { name: 'amount', label: 'Amount ($)', type: 'number', step: '0.01', required: true },
  { name: 'merchant_id', label: 'Merchant ID', type: 'text' },
  { name: 'device_id', label: 'Device ID', type: 'text' },
  { name: 'latitude', label: 'Latitude', type: 'number', step: '0.0001' },
  { name: 'longitude', label: 'Longitude', type: 'number', step: '0.0001' }
];

const Field = ({ name, label, type, step, required, value, onChange }) => (
  <div>
    <label style={FIELD_LABEL_STYLE}>{label}</label>
    <input
      type={type}
      step={step}
      name={name}
      value={value}
      onChange={onChange}
      required={required}
      style={INPUT_STYLE}
      onFocus={(e) => (e.target.style.borderColor = 'rgba(255, 255, 255, 0.35)')}
      onBlur={(e) => (e.target.style.borderColor = 'rgba(255, 255, 255, 0.1)')}
    />
  </div>
);

const App = () => {
  const [form, setForm] = useState({
    transaction_id: crypto.randomUUID(),
    user_id: 'user_' + Math.floor(Math.random() * 1000),
    amount: '',
    merchant_id: 'merchant_' + Math.floor(Math.random() * 100),
    device_id: 'device_' + Math.floor(Math.random() * 100),
    latitude: '40.7128',
    longitude: '-74.0060',
    timestamp: new Date().toISOString()
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload = {
        ...form,
        amount: parseFloat(form.amount),
        latitude: parseFloat(form.latitude),
        longitude: parseFloat(form.longitude),
        timestamp: new Date().toISOString()
      };

      const response = await fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error('Failed to analyze transaction');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  const isFraud = result?.decision === 'FRAUD';

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px',
        background: '#141413'
      }}
    >
      <div style={{ maxWidth: '900px', width: '100%' }}>
        <header style={{ textAlign: 'center', marginBottom: '40px' }}>
          <h1
            style={{
              fontSize: '44px',
              fontWeight: 600,
              marginBottom: '10px',
              color: '#f0f0ee',
              letterSpacing: '-0.5px'
            }}
          >
            Swarm fraud detection
          </h1>
          <p style={{ color: '#8a8a86', fontSize: '17px', fontWeight: 400 }}>
            Real-time transaction analysis using multi-agent consensus
          </p>
        </header>

        <div
          style={{
            background: 'rgba(255, 255, 255, 0.03)',
            borderRadius: '20px',
            padding: '40px',
            border: '1px solid rgba(255, 255, 255, 0.08)'
          }}
        >
          <form onSubmit={handleSubmit}>
            <div style={{ marginBottom: '24px' }}>
              <label style={FIELD_LABEL_STYLE}>Transaction ID</label>
              <input
                type="text"
                name="transaction_id"
                value={form.transaction_id}
                readOnly
                style={{ ...INPUT_STYLE, color: '#b5b5b0' }}
              />
            </div>

            <div style={GRID_STYLE}>
              {FIELDS.slice(0, 2).map((f) => (
                <Field key={f.name} {...f} value={form[f.name]} onChange={handleChange} />
              ))}
            </div>

            <div style={GRID_STYLE}>
              {FIELDS.slice(2, 4).map((f) => (
                <Field key={f.name} {...f} value={form[f.name]} onChange={handleChange} />
              ))}
            </div>

            <div style={{ ...GRID_STYLE, marginBottom: '32px' }}>
              {FIELDS.slice(4, 6).map((f) => (
                <Field key={f.name} {...f} value={form[f.name]} onChange={handleChange} />
              ))}
            </div>

            <button
              type="submit"
              disabled={loading}
              style={{
                width: '100%',
                padding: '18px',
                background: loading ? '#2a2a28' : '#e8e8e4',
                color: loading ? '#8a8a86' : '#141413',
                border: 'none',
                borderRadius: '12px',
                fontSize: '16px',
                fontWeight: 600,
                cursor: loading ? 'not-allowed' : 'pointer',
                transition: 'opacity 0.2s ease'
              }}
              onMouseOver={(e) => {
                if (!loading) e.target.style.opacity = '0.85';
              }}
              onMouseOut={(e) => {
                if (!loading) e.target.style.opacity = '1';
              }}
            >
              {loading ? 'Analyzing transaction' : 'Analyze transaction'}
            </button>
          </form>

          {error && (
            <div
              style={{
                marginTop: '30px',
                padding: '20px',
                background: 'rgba(226, 75, 74, 0.1)',
                borderLeft: '3px solid #e24b4a',
                borderRadius: '10px',
                color: '#f0999e'
              }}
            >
              Error: {error}
            </div>
          )}

          {result && (
            <div
              style={{
                marginTop: '30px',
                padding: '32px',
                borderRadius: '16px',
                background: isFraud ? 'rgba(226, 75, 74, 0.08)' : 'rgba(99, 153, 34, 0.08)',
                border: isFraud
                  ? '1px solid rgba(226, 75, 74, 0.3)'
                  : '1px solid rgba(99, 153, 34, 0.3)'
              }}
            >
              <h2
                style={{
                  fontSize: '26px',
                  fontWeight: 600,
                  marginBottom: '16px',
                  color: isFraud ? '#f09595' : '#97c459'
                }}
              >
                {result.decision}
              </h2>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                  gap: '16px',
                  marginBottom: '24px'
                }}
              >
                <div
                  style={{
                    padding: '16px',
                    background: 'rgba(255, 255, 255, 0.03)',
                    borderRadius: '10px',
                    border: '1px solid rgba(255, 255, 255, 0.08)'
                  }}
                >
                  <div
                    style={{
                      fontSize: '13px',
                      textTransform: 'uppercase',
                      letterSpacing: '0.5px',
                      color: '#8a8a86',
                      marginBottom: '6px'
                    }}
                  >
                    Final score
                  </div>
                  <div style={{ fontSize: '24px', fontWeight: 600, color: '#f0f0ee' }}>
                    {result.S_final?.toFixed(4)}
                  </div>
                </div>
                <div
                  style={{
                    padding: '16px',
                    background: 'rgba(255, 255, 255, 0.03)',
                    borderRadius: '10px',
                    border: '1px solid rgba(255, 255, 255, 0.08)'
                  }}
                >
                  <div
                    style={{
                      fontSize: '13px',
                      textTransform: 'uppercase',
                      letterSpacing: '0.5px',
                      color: '#8a8a86',
                      marginBottom: '6px'
                    }}
                  >
                    Latency
                  </div>
                  <div style={{ fontSize: '24px', fontWeight: 600, color: '#f0f0ee' }}>
                    {result.latency_ms?.toFixed(2)} ms
                  </div>
                </div>
              </div>

              <div
                style={{
                  padding: '20px',
                  background: 'rgba(255, 255, 255, 0.03)',
                  borderRadius: '12px',
                  border: '1px solid rgba(255, 255, 255, 0.08)'
                }}
              >
                <div
                  style={{
                    fontSize: '13px',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                    color: '#8a8a86',
                    marginBottom: '10px'
                  }}
                >
                  Justification
                </div>
                <p style={{ color: '#c8c8c4', lineHeight: '1.6' }}>{result.justification}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;