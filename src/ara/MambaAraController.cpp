#include "MambaAraController.hpp"
#include <cstring>

MambaPlaybackRenderer::MambaPlaybackRenderer(ARA::PlugIn::DocumentController* documentController) noexcept
: ARA::PlugIn::PlaybackRenderer(documentController)
{
    MambaInferenceEngine::Config config;
    config.modelPath = "models/mamba_rt.onnx";
    config.maxBlockSize = 1024;
    rtEngine_ = std::make_unique<MambaInferenceEngine>(config);
    rtEngine_->init();
}

MambaPlaybackRenderer::~MambaPlaybackRenderer() noexcept = default;

void MambaPlaybackRenderer::willAddPlaybackRegion(ARA::PlugIn::PlaybackRegion* playbackRegion) noexcept
{
    ARA::PlugIn::PlaybackRenderer::willAddPlaybackRegion(playbackRegion);
}

void MambaPlaybackRenderer::didAddPlaybackRegion(ARA::PlugIn::PlaybackRegion* playbackRegion) noexcept
{
    ARA::PlugIn::PlaybackRenderer::didAddPlaybackRegion(playbackRegion);
}

void MambaPlaybackRenderer::willRemovePlaybackRegion(ARA::PlugIn::PlaybackRegion* playbackRegion) noexcept
{
    ARA::PlugIn::PlaybackRenderer::willRemovePlaybackRegion(playbackRegion);
}

void MambaPlaybackRenderer::didRemovePlaybackRegion(ARA::PlugIn::PlaybackRegion* playbackRegion) noexcept
{
    ARA::PlugIn::PlaybackRenderer::didRemovePlaybackRegion(playbackRegion);
}

ARA::PlugIn::Document* MambaDocumentController::doCreateDocument() noexcept
{
    return new ARA::PlugIn::Document(this);
}

void MambaDocumentController::doDestroyDocument(ARA::PlugIn::Document* document) noexcept
{
    delete document;
}

ARA::PlugIn::MusicalContext* MambaDocumentController::doCreateMusicalContext(ARA::PlugIn::Document* document, ARA::ARAMusicalContextHostRef hostRef) noexcept
{
    return new ARA::PlugIn::MusicalContext(document, hostRef);
}

void MambaDocumentController::doDestroyMusicalContext(ARA::PlugIn::MusicalContext* musicalContext) noexcept
{
    delete musicalContext;
}

void MambaDocumentController::doUpdateMusicalContextContent(ARA::PlugIn::MusicalContext*, const ARA::ARAContentTimeRange*, ARA::ContentUpdateScopes) noexcept
{
}

ARA::PlugIn::RegionSequence* MambaDocumentController::doCreateRegionSequence(ARA::PlugIn::Document* document, ARA::ARARegionSequenceHostRef hostRef) noexcept
{
    return new ARA::PlugIn::RegionSequence(document, hostRef);
}

void MambaDocumentController::doDestroyRegionSequence(ARA::PlugIn::RegionSequence* regionSequence) noexcept
{
    delete regionSequence;
}

ARA::PlugIn::AudioSource* MambaDocumentController::doCreateAudioSource(ARA::PlugIn::Document* document, ARA::ARAAudioSourceHostRef hostRef) noexcept
{
    return new ARA::PlugIn::AudioSource(document, hostRef);
}

void MambaDocumentController::doDestroyAudioSource(ARA::PlugIn::AudioSource* audioSource) noexcept
{
    delete audioSource;
}

void MambaDocumentController::doUpdateAudioSourceContent(ARA::PlugIn::AudioSource*, const ARA::ARAContentTimeRange*, ARA::ContentUpdateScopes) noexcept
{
}

void MambaDocumentController::willEnableAudioSourceSamplesAccess(ARA::PlugIn::AudioSource*, bool) noexcept
{
}

void MambaDocumentController::didEnableAudioSourceSamplesAccess(ARA::PlugIn::AudioSource*, bool) noexcept
{
}

ARA::PlugIn::AudioModification* MambaDocumentController::doCreateAudioModification(ARA::PlugIn::AudioSource* audioSource, ARA::ARAAudioModificationHostRef hostRef, const ARA::PlugIn::AudioModification* optionalModificationToClone) noexcept
{
    return new ARA::PlugIn::AudioModification(audioSource, hostRef, optionalModificationToClone);
}

void MambaDocumentController::doDestroyAudioModification(ARA::PlugIn::AudioModification* audioModification) noexcept
{
    delete audioModification;
}

ARA::PlugIn::PlaybackRegion* MambaDocumentController::doCreatePlaybackRegion(ARA::PlugIn::AudioModification* modification, ARA::ARAPlaybackRegionHostRef hostRef) noexcept
{
    return new ARA::PlugIn::PlaybackRegion(modification, hostRef);
}

void MambaDocumentController::doDestroyPlaybackRegion(ARA::PlugIn::PlaybackRegion* playbackRegion) noexcept
{
    delete playbackRegion;
}

ARA::PlugIn::PlaybackRenderer* MambaDocumentController::doCreatePlaybackRenderer() noexcept
{
    return new MambaPlaybackRenderer(this);
}

void MambaDocumentController::doDestroyPlaybackRenderer(ARA::PlugIn::PlaybackRenderer* playbackRenderer) noexcept
{
    delete playbackRenderer;
}

ARA::PlugIn::EditorRenderer* MambaDocumentController::doCreateEditorRenderer() noexcept
{
    return nullptr;
}

void MambaDocumentController::doDestroyEditorRenderer(ARA::PlugIn::EditorRenderer*) noexcept
{
}

ARA::PlugIn::EditorView* MambaDocumentController::doCreateEditorView() noexcept
{
    return nullptr;
}

void MambaDocumentController::doDestroyEditorView(ARA::PlugIn::EditorView*) noexcept
{
}

bool MambaDocumentController::doRestoreObjectsFromArchive(ARA::PlugIn::HostArchiveReader*, const ARA::PlugIn::RestoreObjectsFilter*) noexcept
{
    return true;
}

bool MambaDocumentController::doStoreObjectsToArchive(ARA::PlugIn::HostArchiveWriter*, const ARA::PlugIn::StoreObjectsFilter*) noexcept
{
    return true;
}

const char* MambaFactoryConfig::getFactoryID() const noexcept
{
    return "com.mamba.audio.suite.ara";
}

const char* MambaFactoryConfig::getPlugInName() const noexcept
{
    return "Mamba-SSM Processor (ARA)";
}

const char* MambaFactoryConfig::getManufacturerName() const noexcept
{
    return "Elite DSP";
}

const char* MambaFactoryConfig::getInformationURL() const noexcept
{
    return "https://example.com/mamba_audio_suite";
}

const char* MambaFactoryConfig::getVersion() const noexcept
{
    return "1.0.0";
}

const char* MambaFactoryConfig::getDocumentArchiveID() const noexcept
{
    return "com.mamba.audio.suite.archive";
}

ARA::PlugIn::DocumentController* MambaFactoryConfig::createDocumentController(const ARA::PlugIn::PlugInEntry* entry, const ARA::ARADocumentControllerHostInstance* instance) const noexcept
{
    return new MambaDocumentController(entry, instance);
}

const ARA::ARAFactory* getMambaARAFactory() noexcept
{
    return ARA::PlugIn::PlugInEntry::getPlugInEntry<MambaFactoryConfig, MambaDocumentController>()->getFactory();
}
