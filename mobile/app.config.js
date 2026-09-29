const fs = require('fs');
const { expo } = require('./app.json');

module.exports = () => {
  const config = { ...expo, android: { ...expo.android } };
  const services = process.env.GOOGLE_SERVICES_FILE;
  if (services && fs.existsSync(services)) config.android.googleServicesFile = services;
  const projectId = process.env.EXPO_PUBLIC_EAS_PROJECT_ID;
  if (projectId) config.extra = { eas: { projectId } };
  return config;
};
